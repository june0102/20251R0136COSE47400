import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import build_model
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
from sklearn.model_selection import train_test_split
import time

def filter_valid_rows(df, root_dirs):
    valid = []
    for _, row in df.iterrows():
        file_name = row['file_name'].replace('.mp4', '.avi')
        folder_name = file_name.replace('.avi', '')
        for root_dir in root_dirs:
            vp = os.path.join(root_dir, 'pycrop', folder_name, '00000.avi')
            ap = os.path.join(root_dir, 'aligns', f'{folder_name}.align')
            if os.path.exists(vp) and os.path.exists(ap):
                row['used_root'] = root_dir
                valid.append(row)
                break
    return pd.DataFrame(valid)

def pad_collate(batch):

    batch = [b for b in batch if b is not None]


    if len(batch) == 0:

        return None
    videos, ctc_targets, ctc_lengths, ce_targets = zip(*batch)

    max_len = max(v.size(1) for v in videos)
    padded_videos = []
    for v in videos:
        pad_len = max_len - v.size(1)
        if pad_len > 0:
            pad = torch.zeros((3, pad_len, 96, 96), dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        padded_videos.append(v)
    videos_tensor = torch.stack(padded_videos)
    return videos_tensor, torch.cat(ctc_targets), torch.stack(ctc_lengths), torch.stack(ce_targets)

ce_label_map = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
ctc_label_map = ['<sil>'] + ce_label_map
ce_label_to_idx = {label: i for i, label in enumerate(ce_label_map)}
ctc_label_to_idx = {label: i for i, label in enumerate(ctc_label_map)}

class LipPinDataset(Dataset):
    def __init__(self, root_dirs, label_df):
        self.root_dirs = root_dirs
        self.label_df = label_df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)), transforms.ToTensor()
        ])
        self.video_cache = {}
        self.align_cache = {}

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        file_name = row['file_name'].replace('.mp4', '.avi')
        folder_name = file_name.replace('.avi', '')
        digits = row['digit_words'].split('|')
        used_root = row['used_root']

        if folder_name in self.video_cache:
            frames = self.video_cache[folder_name]
        else:
            video_path = os.path.join(used_root, 'pycrop', folder_name, '00000.avi')
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                f = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                frames.append(self.transform(f))
            cap.release()
            if len(frames) == 0:
                return None
            self.video_cache[folder_name] = frames

        video = torch.stack(frames, dim=1)

        if folder_name in self.align_cache:
            ctc_labels = self.align_cache[folder_name]
        else:
            align_path = os.path.join(used_root, 'aligns', f'{folder_name}.align')
            with open(align_path, 'r') as f:
                lines = f.readlines()
            ctc_labels = [ctc_label_to_idx[line.strip().split()[2]] for line in lines if line.strip().split()[2] in ctc_label_to_idx]
            if len(ctc_labels) == 0:
                return None
            self.align_cache[folder_name] = ctc_labels

        ce_labels = torch.tensor([ce_label_to_idx[d] for d in digits], dtype=torch.long)
        return video, torch.tensor(ctc_labels), torch.tensor(len(ctc_labels)), ce_labels

class HybridCTCDecoder(nn.Module):
    def __init__(self, input_dim=512, rnn_hidden_dim=256, num_classes=11, ce_seq_len=4, alpha=0.3, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.ce_seq_len = ce_seq_len
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.rnn = nn.GRU(input_dim, rnn_hidden_dim, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(rnn_hidden_dim * 2)
        self.ctc_fc = nn.Linear(rnn_hidden_dim * 2, num_classes)
        self.ce_fc = nn.Linear(rnn_hidden_dim * 2, (num_classes - 1) * ce_seq_len)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = self.norm(self.dropout(rnn_out))
        ctc_logits = self.ctc_fc(rnn_out)
        pooled = torch.mean(rnn_out, dim=1)
        ce_logits = self.ce_fc(pooled).view(-1, self.ce_seq_len, self.num_classes - 1)
        return ctc_logits, ce_logits

    def compute_loss(self, ctc_logits, ce_logits, ctc_targets, ctc_lengths, ce_targets):
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full((ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long).to(ctc_logits.device)
        ctc_loss = F.ctc_loss(log_probs, ctc_targets, input_lengths, ctc_lengths, blank=0)
        ce_loss = F.cross_entropy(ce_logits.view(-1, self.num_classes - 1), ce_targets.view(-1), weight=self.class_weights)
        return self.alpha * ctc_loss + (1 - self.alpha) * ce_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dirs = [f'20251R0136COSE47400/4digit_dataset/s{i}' for i in range(1, 7)]
    dfs = [pd.read_csv(p) for p in glob.glob('/home/elicer/20251R0136COSE47400/4digit_dataset/s*/labels.csv')]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = filter_valid_rows(full_df, root_dirs)
    if len(full_df) == 0:
        raise RuntimeError("No valid samples with 00000.avi and align file found.")

    train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42, shuffle=True)
    counter = Counter(chain.from_iterable(train_df['digit_words'].str.split('|')))
    train_df = train_df.copy()
    train_df.loc[:, 'sample_weight'] = train_df.apply(lambda row: 1.0 / (sum([counter[d] for d in row['digit_words'].split('|')]) / 4), axis=1)
    sampler = WeightedRandomSampler(torch.DoubleTensor(train_df['sample_weight'].values), len(train_df))

    train_set = LipPinDataset(root_dirs, train_df)
    val_set = LipPinDataset(root_dirs, val_df)
    train_loader = DataLoader(train_set, batch_size=4, sampler=sampler, collate_fn=pad_collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=pad_collate, num_workers=2, pin_memory=True)

    encoder = build_model('vtp_encoderonly', vocab=512, visual_dim=512)
    encoder.load_state_dict(torch.load('checkpoints/tokenizers/ft_lrs3-2.pth', map_location=device)['state_dict'], strict=False)
    encoder = encoder.to(device)
    for n, p in encoder.named_parameters():
        p.requires_grad = "lora" in n

    digit_freq = [2201, 2159, 2186, 2183, 2166, 2141, 2221, 2277, 2199, 2131]
    weights = torch.tensor([sum(digit_freq) / f for f in digit_freq], dtype=torch.float, device=device)

    decoder = HybridCTCDecoder(num_classes=11, class_weights=weights, alpha=0.3).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters())), lr=1e-4, weight_decay=1e-5)

    train_losses, val_losses = [], []
    for epoch in range(30):
        start_time = time.time()
        encoder.train(); decoder.train(); epoch_losses = []
        first_batch_logged = False
        for batch in train_loader:
            if batch is None:
                continue
            if not first_batch_logged:
                print("✅ Training started: First batch loaded successfully.")
                first_batch_logged = True
            # ✅ tensor를 GPU에 매니저에 복사 발생 없게 전송
            x, ctc_targets, ctc_lengths, ce_targets = batch
            x = x.to(device, non_blocking=True)
            ctc_targets = ctc_targets.to(device, non_blocking=True)
            ctc_lengths = ctc_lengths.to(device, non_blocking=True)
            ce_targets = ce_targets.to(device, non_blocking=True)

            src_mask = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.bool, device=device)

            features = encoder.encode(x, src_mask)
            if isinstance(features, tuple):
                features = features[0]
            ctc_logits, ce_logits = decoder(features)

            loss = decoder.compute_loss(ctc_logits, ce_logits, ctc_targets, ctc_lengths, ce_targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append(loss.item())
        train_losses.append(np.mean(epoch_losses))

        encoder.eval(); decoder.eval(); val_epoch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x, ctc_targets, ctc_lengths, ce_targets = batch
                x = x.to(device, non_blocking=True)
                ctc_targets = ctc_targets.to(device, non_blocking=True)
                ctc_lengths = ctc_lengths.to(device, non_blocking=True)
                ce_targets = ce_targets.to(device, non_blocking=True)

                src_mask = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.bool, device=device)

                features = encoder.encode(x, src_mask)
                if isinstance(features, tuple):
                    features = features[0]
                ctc_logits, ce_logits = decoder(features)
                loss = decoder.compute_loss(ctc_logits, ce_logits, ctc_targets, ctc_lengths, ce_targets)
                val_epoch_losses.append(loss.item())
        val_losses.append(np.mean(val_epoch_losses))

        end_time = time.time()
        print(f"[Epoch {epoch}] Time: {(end_time - start_time):.1f}s | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        # ✅ 매 epoch마다 모델 자동 저장
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'model_epoch_{epoch+1}.pth')


    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Training & Validation Loss'); plt.show()
