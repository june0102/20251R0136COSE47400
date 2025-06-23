import os
from glob import glob

trimmed_dir = r".\trimmed"
frames_dir = r".\frames"
os.makedirs(frames_dir, exist_ok=True)

mp4_files = glob(os.path.join(trimmed_dir, "*.mp4"))

for mp4_path in mp4_files:
    base = os.path.basename(mp4_path)
    if not base.endswith("_trimmed.mp4"):
        continue

    parts = base.replace("_trimmed.mp4", "").split("_")
    if len(parts) != 3:
        print(f"파일명 구조 이상: {base}")
        continue

    speaker = parts[0].lower()  # 's1', 's2', ...
    digit = parts[1]            # '0'~'9'
    vidnum = parts[2]           # '01', '02', ...

    out_dir = os.path.join(frames_dir, speaker, digit, vidnum)
    os.makedirs(out_dir, exist_ok=True)

    # ffmpeg
    out_pattern = os.path.join(out_dir, "frames%03d.png")
    ffmpeg_cmd = f'ffmpeg -y -i "{mp4_path}" "{out_pattern}"'
    print(f"프레임 추출: {base} → {out_pattern}")
    os.system(ffmpeg_cmd)
