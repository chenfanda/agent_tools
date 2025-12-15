import torch
import librosa
import soundfile as sf
import tempfile
import os
import numpy as np
from faster_whisper import WhisperModel  # 👈 替换原版 whisper
from nemo.collections.asr.models import SortformerEncLabelModel
from datetime import timedelta
from scipy.signal import butter, filtfilt

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. Faster-Whisper 模型路径
WHISPER_MODEL_DIR = os.path.join(BASE_DIR, "models", "faster-whisper")
# 2. NeMo 模型路径
NEMO_MODEL_PATH = os.path.join(BASE_DIR, "models", "nemo", "diar_sortformer_4spk-v1.nemo")

def preprocess_audio_for_whisper(audio_input, target_sr=16000):
    """音频预处理"""
    print(f"=== 预处理音频: {audio_input} ===")
    audio, sr = librosa.load(audio_input, sr=target_sr, mono=True)
    print(f"预处理后音频长度: {len(audio)/sr:.2f} 秒")
    return audio, sr

def run_whisper_asr(audio_input, model_size="large-v3"):
    """
    使用 Faster-Whisper 运行 ASR
    """
    print(f"=== 运行 Faster-Whisper ASR (本地模型: {WHISPER_MODEL_DIR}) ===")
    
    # 预处理音频
    audio, sr = preprocess_audio_for_whisper(audio_input, 16000)
    
    # Faster-Whisper 最好接受文件路径
    temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_audio_file.name, audio, sr)
    
    try:
        # === 核心修改: 加载 Faster-Whisper ===
        # compute_type="float16" 是 GPU 加速的关键
        # 如果你的显卡显存小于 8G 且报错，可以改为 "int8"
        model = WhisperModel(WHISPER_MODEL_DIR, device="cuda", compute_type="float32")
        
        print("开始转录...")
        # beam_size=5 是默认推荐
        # vad_filter=True 开启静音过滤，大幅提升速度！
        segments_generator, info = model.transcribe(
            temp_audio_file.name,
            beam_size=5,
            language="zh",
            vad_filter=False,
            word_timestamps=True,
            temperature=0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6
          #  vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"检测语言: {info.language}, 概率: {info.language_probability:.2f}")
        
        # === 关键: 格式转换 ===
        # Faster-Whisper 返回的是对象生成器，我们需要将其转换为
        # 包含字典的列表，以兼容后续的 assign_speakers_to_segments 函数
        segments = []
        for segment in segments_generator:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            
        print(f"检测到句子段落数: {len(segments)}")
        
        # 返回字典结构，保持与原版 whisper 代码兼容
        return {
            "segments": segments,
            "language": info.language
        }
        
    finally:
        if os.path.exists(temp_audio_file.name):
            os.unlink(temp_audio_file.name)

def run_sortformer_diarization(audio_input, model_path=NEMO_MODEL_PATH):
    """运行Sortformer语音角色分离"""
    print(f"=== 运行Sortformer语音角色分离 (模型: {model_path}) ===")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NeMo 模型文件未找到: {model_path}")

    # 使用相同的预处理方法
    audio, sr = preprocess_audio_for_whisper(audio_input, 16000)
    
    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, 16000)
    
    try:
        # 加载模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        diar_model = SortformerEncLabelModel.restore_from(
            restore_path=model_path, 
            map_location=device, 
            strict=False
        )
        diar_model.eval()
        
        # 执行角色分离
        with torch.no_grad():
            predicted_segments = diar_model.diarize(
                audio=temp_file.name,
                batch_size=1
            )
        
        # 解析结果
        segments = []
        if len(predicted_segments) > 0:
            segment_strings = predicted_segments[0]
            for segment_str in segment_strings:
                parts = segment_str.split()
                if len(parts) == 3:
                    segments.append({
                        'start': float(parts[0]),
                        'end': float(parts[1]),
                        'speaker': parts[2]
                    })
        
        segments.sort(key=lambda x: x['start'])
        print(f"检测到说话人段落数: {len(segments)}")
        return segments
        
    finally:
        os.unlink(temp_file.name)

def assign_speakers_to_segments(whisper_result, diarization_segments):
    """给Whisper的每个句子段落分配说话人标签"""
    print("=== 给Whisper句子分配说话人标签 ===")
    
    segments_with_speakers = []
    # 注意：这里 whisper_result['segments'] 现在是我们手动构造的字典列表
    # 所以可以用 ['start'] 访问，代码无需修改
    whisper_segments = whisper_result['segments']

    if not diarization_segments:
        print("⚠️ 警告：没有检测到说话人分段，所有字幕标记为 Speaker 0")
        for seg in whisper_segments:
            segments_with_speakers.append({**seg, 'speaker': 'speaker_0'})
        return segments_with_speakers
    
    for segment in whisper_segments:
        segment_start = segment['start']
        segment_end = segment['end']
        segment_center = (segment_start + segment_end) / 2
        
        # 默认第一个说话人
        best_speaker = diarization_segments[0]['speaker']
        max_overlap = 0.0
        
        for diar_seg in diarization_segments:
            overlap_start = max(segment_start, diar_seg['start'])
            overlap_end = min(segment_end, diar_seg['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg['speaker']
        
        if max_overlap == 0:
            min_distance = float('inf')
            for diar_seg in diarization_segments:
                distance = min(
                    abs(segment_center - diar_seg['start']),
                    abs(segment_center - diar_seg['end'])
                )
                if distance < min_distance:
                    min_distance = distance
                    best_speaker = diar_seg['speaker']
        
        segments_with_speakers.append({
            'start': segment_start,
            'end': segment_end,
            'text': segment['text'],
            'speaker': best_speaker
        })
    
    return segments_with_speakers

def format_time_srt(seconds):
    """将秒数转换为SRT时间格式"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt_with_speakers(segments_with_speakers, output_file):
    """生成带有说话人标记的SRT字幕文件"""
    print(f"=== 生成SRT字幕: {output_file} ===")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments_with_speakers, 1):
            start_time = format_time_srt(segment['start'])
            end_time = format_time_srt(segment['end'])
            speaker_label = segment['speaker'].replace('speaker_', 'Speaker ')
            text_with_speaker = f"[{speaker_label}] {segment['text']}"
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text_with_speaker}\n\n")

def create_simple_srt(whisper_result, output_file):
    """不使用说话人分离时，生成简单的SRT字幕"""
    print(f"=== 生成简单SRT字幕: {output_file} ===")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 注意：这里 whisper_result['segments'] 也是我们构造的字典列表
        for i, segment in enumerate(whisper_result['segments'], 1):
            start_time = format_time_srt(segment['start'])
            end_time = format_time_srt(segment['end'])
            text = segment['text']

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def whisper_with_diarization(audio_input, 
                           whisper_model="large-v3",
                           diar_model_path=NEMO_MODEL_PATH, # 使用新常量
                           output_srt="output_with_speakers.srt"):
    """完整的流程"""
    print(f"处理音频文件: {audio_input}")
    
    # 1. 运行 Faster-Whisper ASR
    whisper_result = run_whisper_asr(audio_input, whisper_model)
    
    # 2. 运行语音角色分离
    # 注意：这里 diar_model_path 默认值已经是新的本地路径了
    diarization_segments = run_sortformer_diarization(audio_input, diar_model_path)
    
    # 3. 给Whisper句子分配说话人标签
    segments_with_speakers = assign_speakers_to_segments(whisper_result, diarization_segments)
    
    # 4. 生成SRT字幕
    create_srt_with_speakers(segments_with_speakers, output_srt)
    
    return segments_with_speakers

if __name__ == "__main__":
    # 测试代码
    pass
