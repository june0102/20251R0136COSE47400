import os
from glob import glob
from pydub import AudioSegment, silence

src_dir = r".\AVDigits"
dst_dir = r".\trimmed"
os.makedirs(dst_dir, exist_ok=True)

mp4_files = glob(os.path.join(src_dir, "**", "*.mp4"), recursive=True)
print(f"총 {len(mp4_files)}개 파일을 처리합니다.")

for i, mp4_path in enumerate(mp4_files, 1):
    basename = os.path.splitext(os.path.relpath(mp4_path, src_dir))[0]
    out_path = os.path.join(dst_dir, basename + "_trimmed.mp4")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(dst_dir, basename + "_temp.wav")
    os.system(f'ffmpeg -y -i "{mp4_path}" -q:a 0 -map a "{wav_path}"')

    try:
        sound = AudioSegment.from_wav(wav_path)
    except Exception as e:
        print(f"{basename}: WAV 읽기 실패 - {e}")
        continue

    nonsilences = silence.detect_nonsilent(sound, min_silence_len=200, silence_thresh=sound.dBFS-16)
    padding = 150

    if nonsilences:
        start = max(nonsilences[0][0] - padding, 0)
        end = min(nonsilences[-1][1] + padding, len(sound))
        start_sec = start / 1000
        end_sec = end / 1000

        cmd = (
            f'ffmpeg -y -i "{mp4_path}" '
            f'-ss {start_sec:.2f} -to {end_sec:.2f} '
            f'-c:v libx264 -preset fast -crf 18 '
            f'-c:a aac -b:a 128k -ac 2 -ar 44100 '
            f'-vf "scale=1920:1080,fps=24" '
            f'"{out_path}"'
        )
        print(f"[{i}/{len(mp4_files)}] {basename} → {out_path}")
        os.system(cmd)
    else:
        print(f"[{i}/{len(mp4_files)}] {basename}: 발화구간 찾지 못함 (건너뜀)")

    try:
        os.remove(wav_path)
    except:
        pass
