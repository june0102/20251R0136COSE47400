#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from config import load_args
from models import build_model
from dataloader import VideoDataset, AugmentationPipeline

def pad_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    videos, ctc_targets, ctc_lengths = zip(*batch)
    max_len = max(v.size(1) for v in videos)
    padded = []
    for v in videos:
        pad_len = max_len - v.size(1)
        if pad_len > 0:
            pad = torch.zeros((3, pad_len, 96, 96), dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        padded.append(v)
    return torch.stack(padded), torch.cat(ctc_targets), torch.tensor(ctc_lengths, dtype=torch.long)

class PinVideoDataset(Dataset):
    def __init__(self, roots, df, args):
        self.df = df.reset_index(drop=True)
        self.reader = VideoDataset(args)
        self.augment = AugmentationPipeline(args)
        self.ctc2idx = {l:i for i,l in enumerate(
            ['<sil>','zero','one','two','three','four',
             'five','six','seven','eight','nine','else']
        )}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = row['file_name'].replace('.mp4','.avi').replace('.avi','')
        rd     = row['root']
        video_path = os.path.join(rd, 'pycrop', folder, '00000.avi')
        align_path = os.path.join(rd, 'aligns', f'{folder}.align')

        frames_np = self.reader.read_video(video_path)
        if frames_np is None or frames_np.size == 0:
            return None

        x = torch.from_numpy(frames_np).unsqueeze(0)  # [1,C,T,H,W]
        x = self.augment(x)
        video = x.squeeze(0)                          # [C,T,H,W]

        try:
            lines = open(align_path).readlines()
            labels = [
                self.ctc2idx.get(ln.strip().split()[2], self.ctc2idx['else'])
                for ln in lines
            ]
        except:
            return None

        if not labels:
            return None

        return video, torch.tensor(labels, dtype=torch.long), len(labels)

def filter_labels_and_paths(args):
    roots = sorted(
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('s')
    )
    dfs = []
    for rd in roots:
        p = os.path.join(rd, 'labels.csv')
        if os.path.isfile(p):
            print(f"📄 Found: {p}")
            df = pd.read_csv(p)
            df['root'] = rd
            dfs.append(df)
        else:
            print(f"⚠️ Missing labels.csv in {rd}")
    if not dfs:
        raise ValueError("❌ labels.csv 파일이 하나도 없습니다.")
    return pd.concat(dfs, ignore_index=True), roots

def main():
    args = load_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    full_df, roots = filter_labels_and_paths(args)
    sampled = []
    for rd in roots:
        df_s = full_df[full_df['root']==rd]
        n    = min(500, len(df_s))
        if n>0:
            sampled.append(df_s.sample(n=n, random_state=42))
    small_df = pd.concat(sampled, ignore_index=True)
    train_df, val_df = train_test_split(small_df, test_size=0.2, random_state=42)

    train_ds = PinVideoDataset(roots, train_df, args)
    val_ds   = PinVideoDataset(roots, val_df, args)
    collate  = lambda b: pad_collate([x for x in b if x is not None])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True
    )

    model = build_model('vtp24x24', vocab=13, visual_dim=512).to(device)
    ckpt  = torch.load(os.path.join(args.checkpoint_dir,'tokenizers/ft_lrs3-2.pth'),
                       map_location=device)
    state = ckpt.get('state_dict', ckpt)
    clean = {k:v for k,v in state.items() if 'lora' not in k}
    model.load_state_dict(clean, strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = ('decoder' in name) or ('generator' in name)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    print("🚀 Training Started...\n")
    for epoch in range(1, args.epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            if batch is None:
                continue
            x, tgt, tgt_len = [b.to(device) for b in batch]
            B, T = x.size(0), x.size(2)

            feats = model.encode(x, torch.ones((B,1,T), device=device, dtype=torch.bool))
            if isinstance(feats, tuple): feats = feats[0]

            raw_out = model.generator(feats)
            if raw_out.dim() == 2:
                V = raw_out.size(1)
                raw_out = raw_out.view(B, T, V)
            out = raw_out  # [B,T,V]

            log_probs = F.log_softmax(out, dim=-1).transpose(0,1)
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            tgt_len = tgt_len.view(-1)

            loss = F.ctc_loss(log_probs, tgt, input_lens, tgt_len, blank=0, zero_infinity=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        elapsed = time.time() - t0
        avg_train = train_loss / len(train_loader)
        print(f"[{epoch}/{args.epochs}] Train Loss: {avg_train:.4f} ⏱ {elapsed:.1f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x, tgt, tgt_len = [b.to(device) for b in batch]
                B, T = x.size(0), x.size(2)

                feats = model.encode(x, torch.ones((B,1,T), device=device, dtype=torch.bool))
                if isinstance(feats, tuple): feats = feats[0]

                raw_out = model.generator(feats)
                if raw_out.dim() == 2:
                    V = raw_out.size(1)
                    raw_out = raw_out.view(B, T, V)
                out = raw_out

                log_probs = F.log_softmax(out, dim=-1).transpose(0,1)
                input_lens = torch.full((B,), T, dtype=torch.long, device=device)
                tgt_len = tgt_len.view(-1)

                val_loss += F.ctc_loss(log_probs, tgt, input_lens, tgt_len,
                                       blank=0, zero_infinity=True).item()

        avg_val = val_loss / len(val_loader)
        print(f"📊 Val Loss: {avg_val:.4f}\n")

        # Checkpoint save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.checkpoint_dir, f'model_ctc_epoch{epoch}.pth'))

if __name__ == '__main__':
    main()
