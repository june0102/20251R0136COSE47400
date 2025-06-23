import os
import cv2
import numpy as np
import random
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# Paths and parameters
frames_root = "/home/elicer/20251R0136COSE47400/frames"
out_root = "/home/elicer/20251R0136COSE47400/5digit_dataset"
speakers = [f"s{i}" for i in range(1, 7)]
digits = [str(d) for d in range(10)]
digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
pin_length = 5        # 5-digit PINs
pins_per_speaker = 300  # Number of videos per speaker

os.makedirs(out_root, exist_ok=True)

def make_pin_pool(pin_length=5, pool_size=300):
    pool = set()
    while len(pool) < pool_size:
        pin = ''.join(random.choices(digits, k=pin_length))
        pool.add(pin)
    return list(pool)

def frames_to_video(frame_list, out_path, fps=24, scale=0.5):
    imgs = [cv2.imread(fp) for fp in frame_list]
    h, w = imgs[0].shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    imgs = [cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) for img in imgs]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (nw, nh))
    for img in imgs:
        out.write(img)
    out.release()

def save_align(align_path, word_list, frame_lens):
    with open(align_path, 'w') as f:
        idx = 0
        for word, flen in zip(word_list, frame_lens):
            if word.lower() == "sil":
                idx += flen
                continue
            f.write(f"{idx} {idx+flen-1} {word}\n")
            idx += flen

def process_speaker(speaker):
    random.seed(speaker + "5d")  # Reproducible for each speaker
    pin_pool = make_pin_pool(pin_length=5, pool_size=300)
    speaker_out = os.path.join(out_root, speaker)
    speaker_align = os.path.join(speaker_out, "aligns")
    os.makedirs(speaker_out, exist_ok=True)
    os.makedirs(speaker_align, exist_ok=True)
    records = []
    print(f"=== Generating 5-digit videos for {speaker} ===")
    for idx, pin in enumerate(tqdm(pin_pool, desc=speaker)):
        digit_list = list(pin)
        word_list = [digit_words[int(d)] for d in digit_list]
        folder_paths = []
        frame_counts = []
        all_frames = []
        digit_video_names = []

        for d in digit_list:
            digit_dir = os.path.join(frames_root, speaker, d)
            if not os.path.exists(digit_dir) or not os.listdir(digit_dir):
                raise RuntimeError(f"Directory not found: {digit_dir}")
            vid_folders = sorted(os.listdir(digit_dir))
            vid_folder = random.choice(vid_folders)
            folder_paths.append(f"{speaker}/{d}/{vid_folder}")
            frame_files = sorted(glob(os.path.join(digit_dir, vid_folder, "frames*.png")))
            all_frames.extend(frame_files)
            frame_counts.append(len(frame_files))
            digit_video_names.append(f"{vid_folder}")

        file_idx = f"{idx+1:03d}"
        pin_word = ''.join(digit_list)
        file_name = f"avdigit_{pin_word}_{file_idx}.mp4"
        out_path = os.path.join(speaker_out, file_name)

        frames_to_video(all_frames, out_path, fps=24, scale=0.5)

        align_path = os.path.join(speaker_align, file_name.replace('.mp4', '.align'))
        save_align(align_path, word_list, frame_counts)

        rec = {
            "file_name": file_name,
            "pin": pin_word,
            "digit_word": '|'.join(word_list),
            "digit_folders": '|'.join(folder_paths),
            "digit_frames": '|'.join(str(fc) for fc in frame_counts),
            "total_frames": sum(frame_counts)
        }
        records.append(rec)

    labels_csv = os.path.join(speaker_out, "labels.csv")
    pd.DataFrame(records).to_csv(labels_csv, index=False)
    print(f"{speaker}: labels.csv and align files saved.")

if __name__ == "__main__":
    with Pool(4) as pool:
        pool.map(process_speaker, speakers)
