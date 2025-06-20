import os
import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader
from config import load_args
from transformers import AutoTokenizer

args = load_args()
tokenizer = AutoTokenizer.from_pretrained(
    'bert-large-uncased', cache_dir='checkpoints/tokenizers',
    bos_token='<bos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>',
    use_fast=True
)

class AugmentationPipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_size = args.img_size
    def center_crop(self, frames):
        crop_x = (frames.size(3) - self.img_size) // 2
        crop_y = (frames.size(4) - self.img_size) // 2
        return frames[:, :, :, crop_x:crop_x + self.img_size, crop_y:crop_y + self.img_size]
    def forward(self, x, flip=False):
        x = x.permute(0, 4, 1, 2, 3)
        faces = self.center_crop(x)
        return faces  # (B, C, T, H, W)

class VideoDataset:
    def __init__(self, args):
        self.frame_size = args.frame_size
        self.normalize = args.normalize
        print('Normalize face:', bool(self.normalize))
    def read_video(self, fpath, start=0, end=None):
        start = max(start - 4, 0)
        end = end + 4 if end else 1e10
        with open(fpath, 'rb') as f:
            vr = VideoReader(f, width=self.frame_size, height=self.frame_size)
            end = min(end, len(vr))
            frames = vr.get_batch(list(range(start, int(end)))).asnumpy().astype(np.float32)
        return frames / 255.0  # (T, H, W, C)

class HybridTrainDataset(Dataset):
    def __init__(self, file_list, args, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.reader = VideoDataset(args)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        video_path = self.file_list[idx]
        frames_np = self.reader.read_video(video_path)
        if frames_np is None or frames_np.size == 0:
            return None
        frames = torch.from_numpy(frames_np)
        if self.transform:
            frames = self.transform(frames.unsqueeze(0)).squeeze(0)
        else:
            frames = frames.permute(0, 3, 1, 2)
        # The following are dummy labels; actual labels come from main script
        ctc_target = torch.randint(0, 12, (10,))
        ctc_len = torch.tensor([len(ctc_target)])
        ce_target = torch.randint(0, 10, (4,))
        return frames, ctc_target, ctc_len, ce_target
