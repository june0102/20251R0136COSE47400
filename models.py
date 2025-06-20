import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderDecoder, PositionwiseFeedForward, PositionalEncoding, EncoderLayer, \
        DecoderLayer, MultiHeadedAttention, Encoder, Decoder, \
        Generator, Embeddings, CTCHeadFFN

from modules import CNN_3d_featextractor, CNN_3d, VTP, VTP_wrapper

def CNN_Baseline(vocab, visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
                    backbone=True):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        (CNN_3d(visual_dim) if backbone else None),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),

        nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Lip-reading VTP model
def VTP_24x24(vocab, visual_dim, N=6, d_model=512, 
                d_ff=2048, h=8, dropout=0.1, backbone=True):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[8, 8], 
                                patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)

    model = EncoderDecoder(
        (VTP_wrapper(CNN_3d_featextractor(d_model, till=24), linear_pooler, in_dim=128, 
                                out_dim=512) if backbone else None),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),

        nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab))
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

###### VSD model
def Silencer_VTP_24x24(visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
                    backbone=False):
    c = copy.deepcopy

    from modules import Silencer
    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[8, 8], 
                            patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)

    model = Silencer(
        (VTP_wrapper(CNN_3d96_featextractor(d_model, till=24), linear_pooler, in_dim=128, 
                                out_dim=512) if backbone else None), c(position),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Linear(d_model, 1))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model




# finetune 추가부분 
def build_model(model_name, **kwargs):
    return builders[model_name](**kwargs)

def VTP_EncoderOnly(vocab, visual_dim, N=6, d_model=512,
                    d_ff=2048, h=8, dropout=0.1, backbone=True):
    c = copy.deepcopy

    from modules import VTP, VTP_wrapper, CNN_3d_featextractor

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[8, 8],
                        patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)

    model = EncoderDecoder(
        (VTP_wrapper(CNN_3d_featextractor(d_model, till=24), linear_pooler, in_dim=128,
                     out_dim=512) if backbone else None),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        None,  # Decoder 제거

        nn.Sequential(c(position)),
        None,  # tgt_embed 제거
        None   # generator 제거
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def VTP_EncoderOnlyCTCFFN(vocab, visual_dim, N=6, d_model=512,
                         d_ff=2048, h=8, dropout=0.1, backbone=True):
    # 1) 기존 encoder‐only 모델 생성
    model = VTP_EncoderOnly(vocab=vocab,
                            visual_dim=visual_dim,
                            N=N, d_model=d_model,
                            d_ff=d_ff, h=h,
                            dropout=dropout,
                            backbone=backbone)
    # 2) CTCHeadFFN 붙이기
    model.ctc_head = CTCHeadFFN(d_model, vocab_size=vocab, d_ff=512, dropout=0.1)
    # 3) head 초기화
    nn.init.xavier_uniform_(model.ctc_head.net[0].weight)
    nn.init.zeros_(model.ctc_head.net[0].bias)
    nn.init.xavier_uniform_(model.ctc_head.net[3].weight)
    nn.init.zeros_(model.ctc_head.net[3].bias)
    return model

# models.py 맨 아래에 추가
def VTP_24x24_CE11(visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, backbone=True):
    """
    VTP_24x24 구조 그대로 쓴 뒤, final Generator만 11-way CE head로 바꾼 버전
    """
    # 기존 VTP_24x24 빌더 호출
    model = VTP_24x24(
        vocab=11,          # <sil>, 0,1,...,9 총 11개
        visual_dim=visual_dim,
        N=N, d_model=d_model,
        d_ff=d_ff, h=h,
        dropout=dropout,
        backbone=backbone
    )
    return model








builders = {
    'cnn_baseline': CNN_Baseline,
    'vtp24x24': VTP_24x24,
    'vtp_encoderonly': VTP_EncoderOnly,  
    'silencer_vtp24x24': Silencer_VTP_24x24,
    'vtp_encoder_ctc_ffn' : VTP_EncoderOnlyCTCFFN,
    'vtp24x24_ce11' : VTP_24x24_CE11
}

