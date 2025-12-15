import os
import subprocess
import torchaudio
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

# === 路径配置 ===
# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 指定 Demucs 模型根目录
DEMUCS_ROOT = os.path.join(BASE_DIR, "models", "demucs")

def extract_audio_from_video(video_path, audio_output_path):
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def separate_audio_tracks(audio_path, output_dir, model_name='htdemucs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # === 关键修改: 设置 TORCH_HOME 环境变量指向本地模型目录 ===
    # Demucs 依赖这个环境变量来寻找模型
    original_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = DEMUCS_ROOT
    
    try:
        print(f"Loading Demucs model from: {DEMUCS_ROOT}")
        model = get_model(name=model_name)
    except Exception as e:
        print(f"Error loading Demucs model: {e}")
        raise
    finally:
        # 恢复环境变量（保持环境整洁）
        if original_torch_home:
            os.environ["TORCH_HOME"] = original_torch_home
        else:
            os.environ.pop("TORCH_HOME", None)

    wav, sr = torchaudio.load(audio_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Demucs inference device: {device}")
    
    sources = apply_model(model, wav[None], device=device)
    sources = sources[0]

    track_names = model.sources
    
    vocals_track = None
    accompaniment_track = None

    for i, name in enumerate(track_names):
        track = sources[i].cpu()
        
        if name == 'vocals':
            vocals_track = track
        else:
            if accompaniment_track is None:
                accompaniment_track = track
            else:
                accompaniment_track += track

    if vocals_track is not None:
        if vocals_track.dim() == 2 and vocals_track.shape[0] != 1:
            vocals_track = vocals_track.mean(dim=0, keepdim=True)
        torchaudio.save(os.path.join(output_dir, "vocals.wav"), vocals_track, sr)

    if accompaniment_track is not None:
        if accompaniment_track.dim() == 2 and accompaniment_track.shape[0] != 1:
            accompaniment_track = accompaniment_track.mean(dim=0, keepdim=True)
        torchaudio.save(os.path.join(output_dir, "accompaniment.wav"), accompaniment_track, sr)

def process_video(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = f"{base}_audio.wav"
    output_dir = f"{base}_demucs_output"

    extract_audio_from_video(video_path, audio_path)
    separate_audio_tracks(audio_path, output_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python audio_refine.py <video_path>")
    else:
        process_video(sys.argv[1])
