#!/usr/bin/env python3
import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import build_model
from finetune_digits_speed import filter_valid_rows, LipPinDataset, pad_collate, HybridCTCDecoder

# --- 상수 정의 ---
ROOT_BASE    = '/home/elicer/20251R0136COSE47400/gridcorpus/archive/pin_dataset'
MODEL_PATH   = '/home/elicer/20251R0136COSE47400/vtp-master/checkpoints/model3_epoch_50.pth'
CE_LABEL_MAP = ['zero','one','two','three','four','five','six','seven','eight','nine']

# 1) 전체 레이블 데이터 로드 & 유효 샘플 필터링
root_dirs = [f'{ROOT_BASE}/s{i}' for i in range(1,25)]
dfs = [pd.read_csv(f) for f in glob.glob(f'{ROOT_BASE}/s*/labels.csv')]
full_df = pd.concat(dfs, ignore_index=True)
full_df = filter_valid_rows(full_df, root_dirs)
if full_df.empty:
    raise RuntimeError("유효한 샘플이 없습니다.")

# 2) 디바이스 및 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder
encoder = build_model('vtp_encoderonly', vocab=512, visual_dim=512).to(device)

# Decoder (학습할 때 사용한 가중치 균형값 그대로)
digit_freq = [4840,5562,5315,5289,5605,5442,5486,5532,5339,5386]
weights = torch.tensor([sum(digit_freq)/f for f in digit_freq], device=device)
decoder = HybridCTCDecoder(num_classes=11, class_weights=weights, alpha=0.4).to(device)

# 체크포인트 로드
ckpt = torch.load(MODEL_PATH, map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
decoder.load_state_dict(ckpt['decoder_state_dict'])
encoder.eval(); decoder.eval()

# 3) 한 세션(s1 또는 s21)에 대해 샘플링 & inference
def run_session(session_id):
    print(f'\n=== Session s{session_id} ===')
    df_s = full_df[full_df['used_root'].str.endswith(f'/s{session_id}')]
    sample_df = df_s.sample(n=10, random_state=42).reset_index(drop=True)
    dataset = LipPinDataset(root_dirs, sample_df)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)

    total_acc = 0.0
    for i, batch in enumerate(loader, 1):
        if batch is None: 
            print(f"  샘플 {i}: 유효하지 않은 데이터, 건너뜀")
            continue
        x, _, _, ce_targets = batch
        x = x.to(device); ce_targets = ce_targets.to(device)

        # 인퍼런스
        src_mask = torch.ones((x.size(0),1,x.size(2)), dtype=torch.bool, device=device)
        features = encoder.encode(x, src_mask)
        if isinstance(features, tuple):
            features = features[0]
        _, ce_logits = decoder(features)

        # 예측 및 정확도 계산
        pred_idxs = ce_logits.argmax(dim=-1).squeeze(0).tolist()
        true_idxs = ce_targets.squeeze(0).tolist()
        preds = [CE_LABEL_MAP[idx] for idx in pred_idxs]
        trues = [CE_LABEL_MAP[idx] for idx in true_idxs]
        seq_acc = sum(p==t for p,t in zip(preds, trues)) / len(trues)
        total_acc += seq_acc

        print(f'  샘플 {i:2d}: 정답={"|".join(trues):15s} 예측={"|".join(preds):15s} 자릿수 정확도={seq_acc*100:5.1f}%')

    print(f'→ s{session_id} 평균 자릿수 정확도: {total_acc/10*100:5.1f}%')

# 4) s1과 s22 실행
if __name__ == '__main__':
    run_session(1)
    run_session(22)
