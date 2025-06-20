#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from collections import Counter
from sklearn.model_selection import train_test_split

from config import load_args
from models import build_model
from dataloader import VideoDataset, AugmentationPipeline

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
        row     = self.df.iloc[idx]
        session = row['session']
        rd      = next(r for r in self.root_dirs if os.path.basename(r)==session)
        fname   = row['file_name'].replace('.mp4','.avi')
        folder  = fname.replace('.avi','')
        video_p = os.path.join(rd, 'pycrop', folder, '00000.avi')
        align_p = os.path.join(rd, 'aligns', f'{folder}.align')

        frames = self.reader.read_video(video_p)
        if frames is None or len(frames)==0:
            return None

        x     = torch.from_numpy(frames).unsqueeze(0)
        x     = self.augment(x)
        video = x.squeeze(0)  # (C, T, H, W)

        with open(align_p,'r') as f:
            lines = f.readlines()
        ctc_lbls = [ self.ctc2idx.get(ln.split()[2],0) for ln in lines ]
        if not ctc_lbls:
            return None

        digits = row['digit_word'].split('|')
        if len(digits)!=4:
            return None
        ce_lbls = [self.ce2idx[d] for d in digits]

        return video, \
               torch.tensor(ctc_lbls, dtype=torch.long), \
               torch.tensor(len(ctc_lbls), dtype=torch.long), \
               torch.tensor(ce_lbls, dtype=torch.long)

def filter_labels_and_paths(args):
    ROOT = args.data_root or args.video_root
    root_dirs = sorted(
        os.path.join(ROOT, d)
        for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and d.startswith('s')
    )
    dfs = []
    for rd in root_dirs:
        p = os.path.join(rd,'labels.csv')
        if os.path.isfile(p):
            df = pd.read_csv(p)
            df['session'] = os.path.basename(rd)
            dfs.append(df)
    if not dfs:
        raise RuntimeError("labels.csv 파일이 없습니다.")
    return pd.concat(dfs,ignore_index=True), root_dirs

def pad_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    videos, ctc_t, ctc_l, ce_t = zip(*batch)
    # pad video to same T
    maxT = max(v.size(1) for v in videos)
    vids = []
    for v in videos:
        C,T,H,W = v.shape
        if T<maxT:
            pad = torch.zeros((C,maxT-T,H,W),dtype=v.dtype)
            v   = torch.cat([v,pad],dim=1)
        vids.append(v)
    vids = torch.stack(vids)                            # (B,C,T_max,H,W)
    ctc_t = torch.cat(ctc_t)                            # flat
    ctc_l = torch.tensor(ctc_l, dtype=torch.long)
    ce_t  = torch.stack(ce_t)                           # (B,4)
    return vids, ctc_t, ctc_l, ce_t

def make_subsequent_mask(N):
    m = torch.triu(torch.ones(N,N,dtype=torch.bool),diagonal=1)
    return ~m  # lower+diag True, future False

def main():
    args   = load_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--data_root {args.data_root}  device={device}")

    # 1) DataFrame & paths
    df, root_dirs = filter_labels_and_paths(args)
    train_df, val_df = train_test_split(df, test_size=0.2,
                                         random_state=42, shuffle=True)

    # 2) CE label balancing
    ce_seqs = train_df['digit_word'].str.split('|').tolist()
    flat    = [d for seq in ce_seqs for d in seq]
    from collections import Counter
    freq    = Counter(flat)
    weights = [1.0/(sum(freq[d] for d in seq)/len(seq)) for seq in ce_seqs]
    sampler = WeightedRandomSampler(
        torch.tensor(weights,dtype=torch.double),
        num_samples=len(weights),
        replacement=True
    )

    # 3) Datasets & Loaders
    train_ds = PinVideoDataset(root_dirs, train_df, args)
    val_ds   = PinVideoDataset(root_dirs, val_df,   args)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              sampler=sampler,
                              collate_fn=pad_collate,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=pad_collate,
                              num_workers=args.num_workers,
                              pin_memory=True)

    # 4) Model & partial load
    model = build_model('vtp24x24_ce11', visual_dim=args.feat_dim)
    model.to(device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir,'tokenizers/ft_lrs3-2.pth'),
                      map_location=device)
    state = ckpt.get('state_dict', ckpt)
    # drop only old generator weights
    state = {k:v for k,v in state.items()
             if not k.startswith('generator.proj')}
    model.load_state_dict(state, strict=False)

    # 5) Freeze / Unfreeze
    for p in model.parameters():
        p.requires_grad = False

    # ─── 반드시 Unfreeze 할 레이어 ────────────────────
    for name, p in model.named_parameters():
        # Transformer decoder
        if name.startswith('decoder'):
            p.requires_grad = True
        # token embedding (tgt_embed.0.lut.weight)
        if name.startswith('tgt_embed.0.lut'):
            p.requires_grad = True
        # final generator
        if name.startswith('generator.proj'):
            p.requires_grad = True

    # Debug: print trainable layers
    print("=== Trainable params ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.shape)
    print("========================")

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    # 6) Teacher-forcing 세팅
    T_tgt    = 4
    tgt_mask = make_subsequent_mask(T_tgt).to(device).unsqueeze(0)
    sil_idx  = 0  # '<sil>' 인덱스

    # 7) Train / Val Loop
    for epoch in range(1, args.epochs+1):
        start = time.time()
        model.train()
        train_loss = 0.0

        for vid, _, _, ce_t in train_loader:
            vid, ce_t = vid.to(device), ce_t.to(device)
            B = vid.size(0)
            src_mask = torch.ones((B,1,vid.size(2)),
                                  device=device, dtype=torch.bool)

            bos    = torch.full((B,1), sil_idx,
                                device=device, dtype=torch.long)
            dec_in = torch.cat([bos, ce_t[:,:-1]], dim=1)  # (B,4)

            logits = model(vid, dec_in, src_mask, tgt_mask)  # (B*4,11)
            loss   = F.cross_entropy(logits,
                                     ce_t.view(-1),
                                     ignore_index=sil_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_tr = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vid, _, _, ce_t in val_loader:
                vid, ce_t = vid.to(device), ce_t.to(device)
                B = vid.size(0)
                src_mask = torch.ones((B,1,vid.size(2)),
                                      device=device, dtype=torch.bool)
                bos    = torch.full((B,1), sil_idx,
                                    device=device, dtype=torch.long)
                dec_in = torch.cat([bos, ce_t[:,:-1]], dim=1)
                logits = model(vid, dec_in, src_mask, tgt_mask)
                val_loss += F.cross_entropy(logits,
                                           ce_t.view(-1),
                                           ignore_index=sil_idx).item()

        avg_val = val_loss / len(val_loader)
        elap    = time.time() - start
        print(f"[{epoch}/{args.epochs}] "
              f"Train CE: {avg_tr:.4f} | Val CE: {avg_val:.4f} ⏱{elap:.1f}s")

        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir,
                                f"vtp24x24_ce4_epoch{epoch}.pth"))

if __name__ == "__main__":
    main()
