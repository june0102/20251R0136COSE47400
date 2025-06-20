#!/usr/bin/env python3.13
# -*- coding: utf-8 -*-
# Requires Python ≥3.13.2

import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from collections import Counter
from sklearn.model_selection import train_test_split

from config import load_args
from models import build_model

# dataloader.py 에 정의된 것들
from dataloader import VideoDataset, AugmentationPipeline

# finetune_digits_speed.py 에 정의된 레이블 필터링·패딩·디코더
from finetune_digits_speed import filter_valid_rows, pad_collate, HybridCTCDecoder

def filter_labels_and_paths(args):
    # -- pin_dataset 최상위 경로 가져오기 --
    if hasattr(args, 'data_root') and args.data_root:
        ROOT = args.data_root
    elif hasattr(args, 'video_root') and args.video_root:
        ROOT = args.video_root
    else:
        ROOT = '20251R0136COSE47400/4digit_dataset'

    root_dirs = [
        os.path.join(ROOT, f's{i}')
        for i in range(1, 7)  
    ]

    # 1) 모든 세션에서 labels.csv 읽기
    dfs = []
    for rd in root_dirs:
        p = os.path.join(rd, 'labels.csv')
        if os.path.isfile(p):
            dfs.append(pd.read_csv(p))
    full_df = pd.concat(dfs, ignore_index=True)
    # 2) 유효 샘플 필터링
    full_df = filter_valid_rows(full_df, root_dirs)
    if full_df.empty:
        raise RuntimeError("No valid samples found.")
    return full_df, root_dirs

class PinVideoDataset(Dataset):
    def __init__(self, root_dirs, label_df, args):
        self.root_dirs = root_dirs
        self.df = label_df.reset_index(drop=True)
        self.reader = VideoDataset(args)
        self.augment = AugmentationPipeline(args)
        self.ce_map = ['zero','one','two','three','four',
                       'five','six','seven','eight','nine']
        self.ce2idx = {l:i for i,l in enumerate(self.ce_map)}
        self.ctc2idx = {l:i for i,l in enumerate(['<sil>'] + self.ce_map)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['file_name'].replace('.mp4','.avi')
        folder = fname.replace('.avi','')
        used_root = row['used_root']
        video_path = os.path.join(used_root, 'pycrop', folder, '00000.avi')
        align_path = os.path.join(used_root, 'aligns', f'{folder}.align')

        # 1) decord로 (T, frame_size, frame_size, 3) 읽고 [0–1]
        frames_np = self.reader.read_video(video_path)
        if frames_np is None or len(frames_np) == 0:
            return None
        # 2) center-crop to img_size & (B, C, T, H, W)
        x = torch.from_numpy(frames_np).unsqueeze(0)  # (1, T, H, W, C)
        x = self.augment(x)                           # (1, C, T, H, W)
        video = x.squeeze(0)                          # (C, T, H, W)

        with open(align_path,'r') as f:
            lines = f.readlines()
        ctc_labels = [
            self.ctc2idx[ln.strip().split()[2]]
            for ln in lines
            if ln.strip().split()[2] in self.ctc2idx
        ]
        if len(ctc_labels)==0:
            return None

        digits = row['digit_word'].split('|')
        ce_labels = [ self.ce2idx[d] for d in digits ]
        if frames_np is None or len(frames_np) == 0:
            print(f"🚨 idx={idx} : 영상 로드 실패 (frames_np None or empty)")
            return None
        if len(ctc_labels)==0:
            print(f"🚨 idx={idx} : align 파일 라벨 없음")
            return None

        return (
            video,
            torch.tensor(ctc_labels, dtype=torch.long),
            torch.tensor(len(ctc_labels), dtype=torch.long),
            torch.tensor(ce_labels, dtype=torch.long)
        )

def main():
    args = load_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    #(원본 VTP pipeline 그대로 사용)
    #VideoDataset(args) → (T,160,160,3) 읽고
     #AugmentationPipeline(args) → center-crop to (T,96,96) & permute
    # 추가 리사이즈/ToTensor는 내부에서 이미 처리됩니다.


    full_df, root_dirs = filter_labels_and_paths(args)

    # 세션당 최대 1000개 샘플링
    sampled = []
    for rd in root_dirs:
        df_s = full_df[full_df['used_root']==rd]
        n = min(500, len(df_s))
        if n>0:
            sampled.append(df_s.sample(n=n, random_state=42))
    small_df = pd.concat(sampled, ignore_index=True)

    train_df, val_df = train_test_split(
        small_df, test_size=0.2, random_state=42, shuffle=True
    )
        # ─── (추가) train/test label CSV로 저장 ────────────────────────────
    # data_root/checkpoint_dir 아래에 labels_subsets 폴더를 만들어 저장
    out_dir = os.path.join(args.checkpoint_dir, "labels_subsets")
    os.makedirs(out_dir, exist_ok=True)
    train_csv = os.path.join(out_dir, "train_label_AVDigits.csv")
    val_csv   = os.path.join(out_dir, "val_labels_AVDigits.csv")
    # full_df 의 used_root 컬럼이 "/.../pin_dataset/sX" 형태이므로 세션 sX 추출
    def extract_session(path):
        return os.path.basename(path)  # "s1", "s2", ...
    train_df = train_df.copy()
    val_df   = val_df.copy()
    train_df["session"] = train_df["used_root"].map(extract_session)
    val_df  ["session"] = val_df  ["used_root"].map(extract_session)
    train_df.to_csv(train_csv, index=False, columns=["session","file_name","digit_word"])
    val_df.to_csv(  val_csv, index=False, columns=["session","file_name","digit_word"])
    print(f"▶ Saved train labels: {train_csv}")
    print(f"▶ Saved  val labels: {val_csv}")

    ce_seqs = train_df['digit_word'].str.split('|').tolist()
    flat    = [d for seq in ce_seqs for d in seq]
    freq    = Counter(flat)
    sample_weights = [
        1.0 / (sum(freq[d] for d in seq)/len(seq))
        for seq in ce_seqs
    ]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_ds = PinVideoDataset(root_dirs, train_df, args)
    val_ds   = PinVideoDataset(root_dirs, val_df,   args)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        collate_fn=pad_collate, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=pad_collate, num_workers=args.num_workers, pin_memory=True
    )

    # 모델 생성 및 로드
    encoder = build_model('vtp_encoderonly', vocab=512, visual_dim=512).to(device)
    PRETRAIN = os.path.join(args.checkpoint_dir, 'tokenizers/ft_lrs3-2.pth')
    ckpt_pre = torch.load(PRETRAIN, map_location=device)
    encoder.load_state_dict(ckpt_pre.get('state_dict',ckpt_pre), strict=False)
    for n,p in encoder.named_parameters():
        p.requires_grad = ('lora' in n)

    digit_freq = [2201, 2159, 2186, 2183, 2166, 2141, 2221, 2277, 2199, 2131]
    cw = torch.tensor([sum(digit_freq)/f for f in digit_freq], device=device)
    decoder = HybridCTCDecoder(num_classes=11, class_weights=cw, alpha=0.3).to(device)

    # ─── Debug: trainable 파라미터 수 확인 ─────────────────────────
    total_enc = sum(p.numel() for p in encoder.parameters())
    train_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_dec = sum(p.numel() for p in decoder.parameters())
    train_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[Debug] Encoder total params: {total_enc:,}")
    print(f"[Debug] Encoder trainable params: {train_enc:,}")
    print(f"[Debug] Decoder total params: {total_dec:,}")
    print(f"[Debug] Decoder trainable params: {train_dec:,}")
    # ─────────────────────────────────────────────────────────────

    optimizer = Adam(
        list(filter(lambda p:p.requires_grad, encoder.parameters())) +
        list(decoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs+1):
        encoder.train(); decoder.train()
        epoch_loss = 0.0
        first_batch = True
        print("=== train_loader 준비 완료, 배치 루프 진입 ===")
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"🚨 batch_idx={batch_idx}: batch is None (pad_collate 문제 또는 모든 샘플 None)")
                continue

            # 첫 배치 shape만 확인
            if first_batch:
                print(f"✅ 첫 batch 로드! batch_idx={batch_idx}")
                for i, b in enumerate(batch):
                    if hasattr(b, "shape"):
                        print(f"  batch[{i}].shape = {b.shape}")
                print(f"✅ Epoch {epoch}: first batch ok", flush=True)
                first_batch = False

            # 실제 학습 진행
            x, ctc_t, ctc_l, ce_t = [b.to(device) for b in batch]
            mask = torch.ones((x.size(0),1,x.size(2)), device=device, dtype=torch.bool)
            feats = encoder.encode(x, mask)
            if isinstance(feats, tuple): feats = feats[0]
            ctc_logits, ce_logits = decoder(feats)

            logp = F.log_softmax(ctc_logits, dim=-1).transpose(0,1)
            in_l = torch.full((x.size(0),), ctc_logits.size(1), device=device, dtype=torch.long)
            ctc_loss = F.ctc_loss(logp, ctc_t, in_l, ctc_l, blank=0)
            ce_loss  = F.cross_entropy(ce_logits.view(-1,10), ce_t.view(-1), weight=decoder.class_weights)
            loss = 0.1*ctc_loss + 0.9*ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)
        encoder.eval(); decoder.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                x, ctc_t, ctc_l, ce_t = [b.to(device) for b in batch]
                mask = torch.ones((x.size(0),1,x.size(2)), device=device, dtype=torch.bool)
                feats = encoder.encode(x, mask)
                if isinstance(feats, tuple): feats = feats[0]
                ctc_logits, ce_logits = decoder(feats)

                logp = F.log_softmax(ctc_logits, dim=-1).transpose(0,1)
                ctc_lens = torch.full((x.size(0),), ctc_logits.size(1), device=device,dtype=torch.long)
                cl = F.ctc_loss(logp, ctc_t, ctc_lens, ctc_l, blank=0)
                cel = F.cross_entropy(ce_logits.view(-1,10), ce_t.view(-1), weight=decoder.class_weights)
                vloss += (0.3*cl + 0.7*cel).item()

        print(f"[{epoch}/{args.epochs}] Train {avg_train:.4f} | Val {vloss/len(val_loader):.4f}", flush=True)

        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(args.checkpoint_dir, f'model_avdigits_finetune4_epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()
