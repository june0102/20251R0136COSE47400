import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import build_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # 🔧 수정된 부분
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

def pad_collate(batch):
    videos, ctc_targets, ctc_lengths, ce_targets = zip(*batch)

    # 비디오 입력 (3, T, 96, 96) → (B, 3, T_max, 96, 96) 로 패딩
    max_len = max(v.size(1) for v in videos)
    padded_videos = []
    for v in videos:
        pad_len = max_len - v.size(1)
        if pad_len > 0:
            pad = torch.zeros((3, pad_len, 96, 96), dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        padded_videos.append(v)
    videos_tensor = torch.stack(padded_videos)

    # CTC target은 이어붙이기만 하면 됨
    ctc_targets_tensor = torch.cat(ctc_targets)
    ctc_lengths_tensor = torch.stack(ctc_lengths)

    # CE target은 고정 길이 (4개)라 그대로 스택
    ce_targets_tensor = torch.stack(ce_targets)

    return videos_tensor, ctc_targets_tensor, ctc_lengths_tensor, ce_targets_tensor


# ✅ 레이블 매핑
label_map = ['<sil>', 'zero', 'one', 'two', 'three', 'four', 'five',
             'six', 'seven', 'eight', 'nine', 'else']
label_to_idx = {label: i for i, label in enumerate(label_map)}

# ✅ 데이터셋 클래스
class LipPinDataset(Dataset):
    def __init__(self, root_dirs, label_source):
        self.root_dirs = root_dirs  # 여러 폴더를 리스트로 받음
        if isinstance(label_source, str):
            self.label_df = pd.read_csv(label_source)
        elif isinstance(label_source, pd.DataFrame):
            self.label_df = label_source.reset_index(drop=True)
        else:
            raise ValueError("label_source must be a file path or a pandas DataFrame.")
        
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        file_name = row['file_name'].replace('.mp4', '.avi')
        folder_name = file_name.replace('.avi', '') 
        digits = row['digit_words'].split('|')
        frame_lengths = list(map(int, row['frame_counts'].split('|')))

        # 여러 폴더에서 파일을 찾음
        video_path = None
        align_path = None
        for root_dir in self.root_dirs:
            potential_video_path = os.path.join(root_dir, 'pycrop', folder_name, '00000.avi')
            potential_align_path = os.path.join(root_dir, f'{folder_name}.align')
            if os.path.exists(potential_video_path) and os.path.exists(potential_align_path):
                video_path = potential_video_path
                align_path = potential_align_path
                break

        if video_path is None or align_path is None:
            raise RuntimeError(f"Cannot find video or align file for: {folder_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted from: {video_path}")

        video = torch.stack(frames, dim=1)  # (3, T, H, W)

        with open(align_path, 'r') as f:
            lines = f.readlines()
        ctc_labels = [label_to_idx.get(line.strip().split()[2], label_to_idx['else']) for line in lines]
        ctc_labels = torch.tensor(ctc_labels, dtype=torch.long)
        ctc_len = torch.tensor(len(ctc_labels), dtype=torch.long)

        ce_labels = torch.tensor([label_to_idx[d] for d in digits], dtype=torch.long)

        return video, ctc_labels, ctc_len, ce_labels


# ✅ 디코더 정의
class HybridCTCDecoder(nn.Module):
    def __init__(self, input_dim=512, rnn_hidden_dim=256, num_classes=12, ce_seq_len=4, alpha=0.2, class_weights = None):
        super(HybridCTCDecoder, self).__init__()
        self.alpha = alpha
        self.ce_seq_len = ce_seq_len
        self.num_classes = num_classes
        self.class_weights = class_weights  # <- 명시적 선언

        self.rnn = nn.GRU(input_dim, rnn_hidden_dim, num_layers=3,
                          batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(rnn_hidden_dim * 2)

        self.ctc_fc = nn.Linear(rnn_hidden_dim * 2, num_classes)
        self.ce_fc = nn.Linear(rnn_hidden_dim * 2, num_classes * ce_seq_len)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = self.norm(self.dropout(rnn_out))
        ctc_logits = self.ctc_fc(rnn_out)
        pooled = torch.mean(rnn_out, dim=1)
        ce_logits = self.ce_fc(pooled).view(-1, self.ce_seq_len, self.num_classes)
        return ctc_logits, ce_logits

    def compute_loss(self, ctc_logits, ce_logits, ctc_targets, ctc_lengths, ce_targets):
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full((ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long).to(ctc_logits.device)
        ctc_loss = F.ctc_loss(log_probs, ctc_targets, input_lengths, ctc_lengths, blank=0)
        #ce_loss = F.cross_entropy(ce_logits.view(-1, self.num_classes), ce_targets.view(-1))
        ce_loss = F.cross_entropy(
            ce_logits.view(-1, self.num_classes),
            ce_targets.view(-1),
            weight=self.class_weights)

        
        return self.alpha * ctc_loss + (1 - self.alpha) * ce_loss

# ✅ 디코딩 함수
def decode_ce_logits(ce_logits, label_map):
    pred_indices = ce_logits.argmax(dim=-1)
    pred_labels = [[label_map[idx.item()] for idx in sequence] for sequence in pred_indices]
    return pred_labels

# ✅ 학습 루프
def train_step(encoder, decoder, batch, optimizer, device):
    encoder.train()
    decoder.train()
    x, ctc_targets, ctc_lengths, ce_targets = [b.to(device) for b in batch]
    T = x.size(2)
    src_mask = torch.arange(T, device=device).unsqueeze(0) < ctc_lengths.unsqueeze(1)  # (B, T)
    src_mask = src_mask.unsqueeze(1)  # (B, 1, T)
    features = encoder.encode(x, src_mask=src_mask)
    if isinstance(features, tuple):
        features = features[0]
    ctc_logits, ce_logits = decoder(features)
    loss = decoder.compute_loss(ctc_logits, ce_logits, ctc_targets, ctc_lengths, ce_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ✅ 평가 함수
def evaluate(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, _, _, ce_targets in dataloader:
            x, ce_targets = x.to(device), ce_targets.to(device)
            src_mask = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.bool).to(device)
            features = encoder.encode(x, src_mask=src_mask)
            if isinstance(features, tuple):
                features = features[0]
            _, ce_logits = decoder(features)
            pred = ce_logits.argmax(dim=-1)
            for p, t in zip(pred, ce_targets):
                print("Pred:", [label_map[i] for i in p.cpu().tolist()])
                print("True:", [label_map[i] for i in t.cpu().tolist()])
            # 수정: 위치별 정확도도 출력
            match = (pred == ce_targets)
            correct += match.sum().item()
            total += match.numel()  # 전체 예측 개수
            
    acc = correct / total * 100
    print(f"✅ Test Accuracy: {acc:.2f}%")
    return acc
'''
# ✅ 실행
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_data_root = '/home/elicer/20251R0136COSE47400/gridcorpus/archive/video_sample'
    root_dirs = [os.path.join(base_data_root, f's{i}') for i in range(1, 21)]
    all_csvs = glob.glob(os.path.join(base_data_root, 's*/labels.csv'))
    dfs = []
    for csv_path in all_csvs:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)



from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0

for train_idx, val_idx in kf.split(full_df):
    print(f"\n📂 Fold {fold + 1}/5")

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = LipPinDataset(root_dirs, train_df)
    val_dataset = LipPinDataset(root_dirs, val_df)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=pad_collate)

    encoder = build_model('vtp_encoderonly', vocab=512, visual_dim=512)
    checkpoint = torch.load('checkpoints/tokenizers/ft_lrs3-2.pth', map_location=device)
    encoder.load_state_dict(checkpoint["state_dict"], strict=False)
    encoder = encoder.to(device)
    for name, param in encoder.named_parameters():
        param.requires_grad = "lora" in name

    decoder = HybridCTCDecoder().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters())),
        lr=1e-4, weight_decay = 1e-5
    )

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(30):
        epoch_losses = []
        correct_train = total_train = 0
        pos_correct = [0] * 4
        pos_total = [0] * 4

        for batch in train_loader:
            loss = train_step(encoder, decoder, batch, optimizer, device)
            epoch_losses.append(loss)

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        val_loss = 0
        correct_val = total_val = 0
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            for x, ctc_targets, ctc_lengths, ce_targets in val_loader:
                
                x, ce_targets = x.to(device), ce_targets.to(device)
                src_mask = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.bool).to(device)
                features = encoder.encode(x, src_mask=src_mask)
                if isinstance(features, tuple):
                    features = features[0]
                ctc_logits, ce_logits = decoder(features)
                pred = ce_logits.argmax(dim=-1)


                # ✅ 디코딩 결과와 정답 출력---------------------
                for i in range(min(2, pred.size(0))):  # 배치 내 2개만 확인
                    pred_tokens = pred[i].cpu().tolist()
                    true_tokens = ce_targets[i].cpu().tolist()

                    print(f"[Sample {i}]")
                    print("🔹 Pred indices :", pred_tokens)
                    print("🔹 True indices :", true_tokens)
                    print("🔹 Pred labels  :", [label_map[idx] for idx in pred_tokens])
                    print("🔹 True labels  :", [label_map[idx] for idx in true_tokens])
                
                # ✅ Softmax 확률 분포 보기 (클래스별로 평균값 확인)
                probs = F.softmax(ce_logits, dim=-1)  # (B, 4, num_classes)
                avg_probs = probs.mean(dim=(0, 1))    # 클래스별 평균 확률
                print("📊 Softmax 평균 분포 (클래스별):")
                for i, p in enumerate(avg_probs.cpu().tolist()):
                    print(f"{label_map[i]:<6}: {p:.4f}")

                # 3. 로짓 출력 보기
                print("🧮 ce_logits raw (첫 샘플):")
                print(ce_logits[0].detach().cpu())  # (4, num_classes)
                #----------------------------------디버깅 코드 --------------
                 # 자리별 accuracy 계산
                for i in range(4):
                    pos_correct[i] += (pred[:, i] == ce_targets[:, i]).sum().item()
                    pos_total[i] += pred.size(0)
                
                match = (pred == ce_targets)
                correct_val += match.sum().item()
                total_val += match.numel()
                
                val_loss += F.cross_entropy(ce_logits.view(-1, decoder.num_classes), ce_targets.view(-1)).item()



        acc = correct_val / total_val * 100
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(acc)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc:.2f}%")
        for i in range(4):
            pos_acc = pos_correct[i] / pos_total[i] * 100
            print(f"  - Position {i+1} Accuracy: {pos_acc:.2f}%")
    # ✅ 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'Fold {fold + 1} Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title(f'Fold {fold + 1} Accuracy')

    plt.tight_layout()
    plt.show()

    fold += 1
'''
# ✅ 실행
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_data_root = '/home/elicer/20251R0136COSE47400/gridcorpus/archive/video_sample'
    root_dirs = [os.path.join(base_data_root, f's{i}') for i in range(1, 21)]
    all_csvs = glob.glob(os.path.join(base_data_root, 's*/labels.csv'))
    dfs = [pd.read_csv(csv_path) for csv_path in all_csvs]
    full_df = pd.concat(dfs, ignore_index=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0

    for train_idx, val_idx in kf.split(full_df):
        print(f"\n\U0001F4C2 Fold {fold + 1}/5")

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # 🔧 [추가] digit 단어 빈도 계산 후 샘플 가중치 계산
        digit_counter = Counter(chain.from_iterable(train_df['digit_words'].str.split('|')))
        train_df['sample_weight'] = train_df.apply(
            lambda row: 1.0 / (sum([digit_counter[d] for d in row['digit_words'].split('|')]) / 4), axis=1
        )
        weights = torch.DoubleTensor(train_df['sample_weight'].values)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_dataset = LipPinDataset(root_dirs, train_df)
        val_dataset = LipPinDataset(root_dirs, val_df)

        # 🔧 shuffle=False + sampler 적용
        train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=2, collate_fn=pad_collate)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=pad_collate)

        encoder = build_model('vtp_encoderonly', vocab=512, visual_dim=512)
        checkpoint = torch.load('checkpoints/tokenizers/ft_lrs3-2.pth', map_location=device)
        encoder.load_state_dict(checkpoint["state_dict"], strict=False)
        encoder = encoder.to(device)
        for name, param in encoder.named_parameters():
            param.requires_grad = "lora" in name

        #decoder = HybridCTCDecoder().to(device)
        # zero ~ nine에 대한 실제 빈도 기반 가중치
        digit_freq = [1320, 1716, 1188, 1650, 1320, 1089, 1155, 1023, 1584, 1155]
        digit_weights = [sum(digit_freq) / f for f in digit_freq]

        # 전체 12개 클래스 가중치로 확장 (<sil>, zero~nine, else)
        class_weights = [1.0] + digit_weights + [1.0]  # index 0: <sil>, index 11: else
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # decoder 정의 시 인자로 넘기기
        decoder = HybridCTCDecoder(class_weights=class_weights, alpha=0.4).to(device)
        

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters())),
            lr=1e-4, weight_decay=1e-5
        )

        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(30):
            epoch_losses = []
            pos_correct = [0] * 4
            pos_total = [0] * 4

            for batch in train_loader:
                loss = train_step(encoder, decoder, batch, optimizer, device)
                epoch_losses.append(loss)

            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)

            val_loss = 0
            correct_val = total_val = 0
            encoder.eval()
            decoder.eval()

            batch_debug_limit = 10  # ✅ 추가
            batch_debug_count = 0   # ✅ 추가

            with torch.no_grad():
                for x, ctc_targets, ctc_lengths, ce_targets in val_loader:
                    x, ce_targets = x.to(device), ce_targets.to(device)
                    src_mask = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.bool).to(device)
                    features = encoder.encode(x, src_mask=src_mask)
                    if isinstance(features, tuple):
                        features = features[0]
                    ctc_logits, ce_logits = decoder(features)
                    pred = ce_logits.argmax(dim=-1)

                    # ✅ 디코딩 예시와 softmax 분포는 10개 배치까지만 출력
                    if batch_debug_count < batch_debug_limit:
                        for i in range(min(2, pred.size(0))):
                            print(f"[Sample {i}]")
                            print("\U0001F539 Pred labels  :", [label_map[idx] for idx in pred[i].cpu().tolist()])
                            print("\U0001F539 True labels  :", [label_map[idx] for idx in ce_targets[i].cpu().tolist()])

                        probs = F.softmax(ce_logits, dim=-1)
                        avg_probs = probs.mean(dim=(0, 1))
                        print("\U0001F4CA Softmax 평균 분포 (클래스별):")
                        for i, p in enumerate(avg_probs.cpu().tolist()):
                            print(f"{label_map[i]:<6}: {p:.4f}")

                        batch_debug_count += 1

                    

                    for i in range(4):
                        pos_correct[i] += (pred[:, i] == ce_targets[:, i]).sum().item()
                        pos_total[i] += pred.size(0)

                    match = (pred == ce_targets)
                    correct_val += match.sum().item()
                    total_val += match.numel()

                    val_loss += F.cross_entropy(ce_logits.view(-1, decoder.num_classes), ce_targets.view(-1)).item()

            acc = correct_val / total_val * 100
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(acc)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc:.2f}%")
            for i in range(4):
                pos_acc = pos_correct[i] / pos_total[i] * 100
                print(f"  - Position {i+1} Accuracy: {pos_acc:.2f}%")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title(f'Fold {fold + 1} Loss')

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.legend()
        plt.title(f'Fold {fold + 1} Accuracy')

        plt.tight_layout()
        plt.show()

        fold += 1