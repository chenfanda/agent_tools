import os
import uuid
import torch
import soundfile as sf
import logging
import logging.config
from datetime import datetime, timedelta
import platform
from pathlib import Path
from typing import Optional, Union, List, Dict
import tempfile
import shutil
import json
import asyncio
import threading
import time
import re
import gc

from pydantic import BaseModel, validator
from cli.SparkTTS import SparkTTS

import librosa
import soundfile as sf
import numpy as np


class TaskFileTracker:
    """
    任务级临时文件追踪器。
    用于记录当前任务产生的所有临时文件路径，并在任务结束时统一清理。
    """
    def __init__(self):
        self.files = []

    def register(self, file_path):
        """注册一个需要清理的文件路径"""
        if file_path and file_path not in self.files:
            self.files.append(file_path)
        return file_path

    def cleanup(self):
        """清理所有注册的文件"""
        logger = logging.getLogger(__name__)
        cleaned_count = 0
        for path in self.files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"清理临时文件失败 {path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"本次任务清理了 {cleaned_count} 个临时文件")
# 日志配置 (保持原有配置)
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8",
        },
        "access_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "access",
            "filename": "logs/access.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console", "file", "error_file"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["access_file"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "ERROR",
            "handlers": ["error_file", "console"],
            "propagate": False,
        },
        "fastapi": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

def setup_logging():
    """设置日志"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.config.dictConfig(LOG_CONFIG)

# 全局变量
model_instance = None
model_last_used = None
# model_lock = threading.Lock()
unload_thread = None
IDLE_TIME_MINUTES = 300
# model_needs_reload = False  # 新增：标识模型是否需要重新加载

model_config = {
    "model_dir": "pretrained_models/Spark-TTS-0.5B",
    "save_dir": "api_results",
    "device": 0
}

# 角色库配置
VOICE_CHARACTERS_CONFIG = {
    "speaker_voice_dir": "speaker_voice",
    "prompt_texts_dir": "prompt_texts",
    "character_avatars_dir": "character_avatars"
}

# 用户自定义录音配置
USER_CUSTOM_VOICES_CONFIG = {
    "base_dir": "user_custom_voices",
    "audio_subdir": "audio",
    "texts_subdir": "texts",
    "metadata_file": "metadata.json"
}

class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str
    prompt_text: Optional[str] = None
    gender: Optional[str] = None
    pitch: Optional[str] = None
    speed: Optional[str] = None

class TTSWithCharacterRequest(BaseModel):
    """使用角色库的TTS请求模型"""
    text: str
    character_id: str  # 角色ID
    gender: Optional[str] = None
    pitch: Optional[str] = None
    speed: Optional[str] = None

class VoiceCharacter(BaseModel):
    """语音角色模型"""
    character_id: str
    name: str
    description: Optional[str] = None
    audio_file: str
    prompt_text: str
    avatar_url: Optional[str] = None
    gender: Optional[str] = None
    language: Optional[str] = None

class TTSResponse(BaseModel):
    """TTS响应模型"""
    message: str
    audio_id: str
    audio_path: str
    timestamp: str
    character_used: Optional[str] = None
    

class SmartSubItem(BaseModel):
    """前端传来的原始字幕项 (毫秒)"""
    id: str
    start_time: float 
    end_time: float   
    text: str         
    new_text: str     
    speaker: str      

class MergedUnit(BaseModel):
    """合并后的生成单元"""
    speaker: str
    text_merged: str       
    prompt_text: str      
    start_time_ms: float   
    end_time_ms: float     
    best_prompt_start: float 
    best_prompt_end: float

class SmartDubbingRequest(BaseModel):
    """智能配音请求"""
    subtitles: List[SmartSubItem]
    original_audio_path: str 
    output_filename: Optional[str] = None
    merge_threshold_ms: float = 500.0 

class SaveCustomVoiceRequest(BaseModel):
    """保存用户自定义录音请求"""
    username: str
    voice_name: str
    text: str
    description: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]{1,50}$', v):
            raise ValueError('用户名只能包含字母、数字和下划线，长度1-50字符')
        return v
    
    @validator('voice_name')
    def validate_voice_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fff\s]{1,100}$', v):
            raise ValueError('录音名称只能包含字母、数字、下划线、中文和空格，长度1-100字符')
        return v

class UserCustomVoice(BaseModel):
    """用户自定义录音信息"""
    voice_id: str
    username: str
    voice_name: str
    text: str
    description: Optional[str] = None
    audio_file_path: str
    created_time: str
    file_size: int
    duration: Optional[float] = None

class SaveCustomVoiceResponse(BaseModel):
    """保存自定义录音响应"""
    message: str
    voice_id: str
    username: str
    voice_name: str
    created_time: str



def unload_model():
    """卸载模型并释放显存"""
    global model_instance, model_needs_reload  # 添加 model_needs_reload
    # with model_lock:
    if model_instance is not None:
        logger = logging.getLogger(__name__)
        logger.info("卸载模型，释放显存...")
        del model_instance
        model_instance = None
        # model_needs_reload = True  # 新增：标记需要重新加载
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("模型已卸载")


def model_unload_checker():
    """检查模型空闲时间的后台线程"""
    global model_instance, model_last_used
    while True:
        time.sleep(60)  # 每分钟检查一次
        # with model_lock:
        if model_instance is not None and model_last_used is not None:
            idle_time = datetime.now() - model_last_used
            if idle_time > timedelta(minutes=IDLE_TIME_MINUTES):
                unload_model()

def update_model_usage():
    """更新模型使用时间"""
    global model_last_used
    model_last_used = datetime.now()

def load_voice_characters() -> Dict[str, VoiceCharacter]:
    """加载语音角色库"""
    logger = logging.getLogger(__name__)
    characters = {}
    
    speaker_voice_dir = VOICE_CHARACTERS_CONFIG["speaker_voice_dir"]
    prompt_texts_dir = VOICE_CHARACTERS_CONFIG["prompt_texts_dir"]
    
    if not os.path.exists(speaker_voice_dir) or not os.path.exists(prompt_texts_dir):
        logger.warning(f"角色库目录不存在: {speaker_voice_dir} 或 {prompt_texts_dir}")
        return characters
    
    # 获取所有音频文件
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a']:
        audio_files.extend(Path(speaker_voice_dir).glob(f"*{ext}"))
    
    for audio_file in audio_files:
        # 获取不带扩展名的文件名作为角色ID
        character_id = audio_file.stem
        
        # 查找对应的提示文本文件
        prompt_file = Path(prompt_texts_dir) / f"{character_id}.txt"
        
        if prompt_file.exists():
            try:
                # 读取提示文本
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                
                # 检查是否有角色配置文件
                config_file = Path(prompt_texts_dir) / f"{character_id}.json"
                character_info = {}
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        character_info = json.load(f)
                
                # 创建角色对象
                character = VoiceCharacter(
                    character_id=character_id,
                    name=character_info.get('name', character_id),
                    description=character_info.get('description'),
                    audio_file=str(audio_file),
                    prompt_text=prompt_text,
                    avatar_url=character_info.get('avatar_url'),
                    gender=character_info.get('gender'),
                    language=character_info.get('language', 'zh')
                )
                
                characters[character_id] = character
                logger.info(f"已加载语音角色: {character_id}")
                
            except Exception as e:
                logger.error(f"加载角色 {character_id} 失败: {str(e)}")
        else:
            logger.warning(f"角色 {character_id} 缺少提示文本文件: {prompt_file}")
    
    logger.info(f"总共加载了 {len(characters)} 个语音角色")
    return characters
	
def initialize_model():
    """初始化TTS模型"""
    global model_instance, model_needs_reload  # 添加 model_needs_reload
    logger = logging.getLogger(__name__)
    
    # with model_lock:
    # 如果模型存在且不需要重新加载，直接返回
    if model_instance is not None :
        update_model_usage()
        return model_instance
    
    # 需要加载模型的情况：模型不存在 或 需要重新加载
    try:
        logger.info(f"Loading model from: {model_config['model_dir']}")
        os.makedirs(model_config['save_dir'], exist_ok=True)
        
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = torch.device(f"mps:{model_config['device']}")
            logger.info(f"Using MPS device: {device}")
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{model_config['device']}")
            logger.info(f"Using CUDA device: {device}")
        else:
            device = torch.device("cpu")
            logger.info("GPU acceleration not available, using CPU")
        
        model_instance = SparkTTS(model_config['model_dir'], device)
        model_needs_reload = False  # 新增：重置标志
        update_model_usage()
        logger.info("Model loaded successfully")
        
        return model_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise Exception(f"Model initialization failed: {str(e)}")

def validate_parameters(gender: Optional[str], pitch: Optional[str], speed: Optional[str]):
    """验证参数"""
    valid_genders = ["male", "female"]
    valid_pitches = ["very_low", "low", "moderate", "high", "very_high"]
    valid_speeds = ["very_low", "low", "moderate", "high", "very_high"]
    
    if gender and gender not in valid_genders:
        raise ValueError(f"Invalid gender. Must be one of: {valid_genders}")
    
    if pitch and pitch not in valid_pitches:
        raise ValueError(f"Invalid pitch. Must be one of: {valid_pitches}")
    
    if speed and speed not in valid_speeds:
        raise ValueError(f"Invalid speed. Must be one of: {valid_speeds}")

def preprocess_audio(audio_path: str, target_sr: int = 16000) -> str:
    """预处理音频文件，确保格式兼容"""
    logger = logging.getLogger(__name__)
    try:
        # 加载音频，自动转换为单声道和目标采样率
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        original_len = len(audio)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        logger.info(f"去除静音: 采样点 {original_len} -> {len(audio)}")

        # 检查音频长度
        duration = len(audio) / sr
        if duration < 1.0:
            raise ValueError(f"音频太短: {duration:.2f}秒，建议至少1秒")

        # 如果音频太长，截取前30秒
        if duration > 30.0:
            audio = audio[:target_sr * 30]
            logger.info(f"音频太长({duration:.2f}秒)，截取前30秒")

        # 归一化音频
        audio = audio / np.max(np.abs(audio)) * 0.9

        # 创建临时处理后的文件
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / f"processed_{Path(audio_path).name}"
        sf.write(str(temp_path), audio, target_sr)

        logger.info(f"音频预处理完成: {audio_path} -> {temp_path}")
        logger.info(f"处理后: 时长{len(audio)/sr:.2f}秒, 采样率{sr}Hz, 单声道")

        return str(temp_path)

    except Exception as e:
        logger.error(f"音频预处理失败 {audio_path}: {str(e)}")
        raise e

# 用户自定义录音相关函数
def validate_username_format(username: str) -> str:
    """验证并清理用户名格式"""
    if not re.match(r'^[a-zA-Z0-9_]{1,50}$', username):
        raise ValueError('用户名只能包含字母、数字和下划线，长度1-50字符')
    return username.strip()

def get_user_custom_dir(username: str) -> Path:
    """获取用户自定义录音目录"""
    username = validate_username_format(username)
    base_dir = Path(USER_CUSTOM_VOICES_CONFIG["base_dir"])
    user_dir = base_dir / username
    return user_dir

def ensure_user_directories(username: str) -> tuple[Path, Path, Path]:
    """确保用户目录存在，返回(用户目录, 音频目录, 文本目录)"""
    user_dir = get_user_custom_dir(username)
    audio_dir = user_dir / USER_CUSTOM_VOICES_CONFIG["audio_subdir"]
    texts_dir = user_dir / USER_CUSTOM_VOICES_CONFIG["texts_subdir"]
    
    # 创建目录
    user_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    texts_dir.mkdir(exist_ok=True)
    
    return user_dir, audio_dir, texts_dir

def get_user_metadata(username: str) -> dict:
    """获取用户元数据"""
    user_dir = get_user_custom_dir(username)
    metadata_file = user_dir / USER_CUSTOM_VOICES_CONFIG["metadata_file"]
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"读取用户 {username} 元数据失败: {e}")
            return {"voices": {}, "created_time": datetime.now().isoformat()}
    else:
        return {"voices": {}, "created_time": datetime.now().isoformat()}

def save_user_metadata(username: str, metadata: dict):
    """保存用户元数据"""
    user_dir = get_user_custom_dir(username)
    metadata_file = user_dir / USER_CUSTOM_VOICES_CONFIG["metadata_file"]
    
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"保存用户 {username} 元数据失败: {e}")
        raise

def generate_voice_id() -> str:
    """生成录音ID"""
    return str(uuid.uuid4())[:8]

def get_custom_voices_statistics() -> dict:
    """获取用户自定义录音统计信息"""
    base_dir = Path(USER_CUSTOM_VOICES_CONFIG["base_dir"])
    total_custom_voices = 0
    total_users = 0
    
    if base_dir.exists():
        for user_dir in base_dir.iterdir():
            if user_dir.is_dir():
                total_users += 1
                metadata = get_user_metadata(user_dir.name)
                total_custom_voices += len(metadata.get("voices", {}))
    
    return {
        "total_users": total_users,
        "total_custom_voices": total_custom_voices
    }

def save_custom_voice_logic(
    username: str,
    voice_name: str,
    text: str,
    description: Optional[str],
    processed_audio_path: str
) -> SaveCustomVoiceResponse:
    """保存用户自定义录音的核心逻辑"""
    logger = logging.getLogger(__name__)
    
    try:
        # 确保用户目录存在
        user_dir, audio_dir, texts_dir = ensure_user_directories(username)
        
        # 生成录音ID和文件名
        voice_id = generate_voice_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"{timestamp}_{voice_id}_{voice_name}.wav"
        text_filename = f"{timestamp}_{voice_id}_{voice_name}.txt"
        
        # 保存音频文件
        final_audio_path = audio_dir / audio_filename
        shutil.copy2(processed_audio_path, final_audio_path)
        
        # 保存文本文件
        text_file_path = texts_dir / text_filename
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # 获取音频文件信息
        audio_stat = final_audio_path.stat()
        
        # 计算音频时长
        try:
            duration = librosa.get_duration(path=str(final_audio_path))
        except Exception:
            duration = None
        
        # 更新用户元数据
        metadata = get_user_metadata(username)
        
        voice_info = {
            "voice_id": voice_id,
            "voice_name": voice_name,
            "text": text,
            "description": description,
            "audio_filename": audio_filename,
            "text_filename": text_filename,
            "created_time": datetime.now().isoformat(),
            "file_size": audio_stat.st_size,
            "duration": duration
        }
        
        metadata["voices"][voice_id] = voice_info
        metadata["updated_time"] = datetime.now().isoformat()
        
        save_user_metadata(username, metadata)
        
        logger.info(f"用户 {username} 成功保存自定义录音: {voice_name} (ID: {voice_id})")
        
        return SaveCustomVoiceResponse(
            message="自定义录音保存成功",
            voice_id=voice_id,
            username=username,
            voice_name=voice_name,
            created_time=voice_info["created_time"]
        )
        
    except Exception as e:
        logger.error(f"保存自定义录音失败: {str(e)}")
        raise

def get_user_custom_voices_logic(username: str) -> List[UserCustomVoice]:
    """获取用户自定义录音列表的核心逻辑"""
    username = validate_username_format(username)
    user_dir = get_user_custom_dir(username)
    
    if not user_dir.exists():
        return []
    
    metadata = get_user_metadata(username)
    voices = []
    
    for voice_id, voice_info in metadata.get("voices", {}).items():
        audio_path = user_dir / USER_CUSTOM_VOICES_CONFIG["audio_subdir"] / voice_info["audio_filename"]
        
        # 检查文件是否仍然存在
        if audio_path.exists():
            voices.append(UserCustomVoice(
                voice_id=voice_id,
                username=username,
                voice_name=voice_info["voice_name"],
                text=voice_info["text"],
                description=voice_info.get("description"),
                audio_file_path=str(audio_path),
                created_time=voice_info["created_time"],
                file_size=voice_info["file_size"],
                duration=voice_info.get("duration")
            ))
    
    # 按创建时间降序排序
    voices.sort(key=lambda x: x.created_time, reverse=True)
    
    return voices

def delete_user_custom_voice_logic(username: str, voice_id: str) -> dict:
    """删除用户自定义录音的核心逻辑"""
    logger = logging.getLogger(__name__)
    
    username = validate_username_format(username)
    user_dir = get_user_custom_dir(username)
    
    if not user_dir.exists():
        raise ValueError("用户不存在")
    
    metadata = get_user_metadata(username)
    
    if voice_id not in metadata.get("voices", {}):
        raise ValueError("录音不存在")
    
    voice_info = metadata["voices"][voice_id]
    
    # 删除音频文件
    audio_path = user_dir / USER_CUSTOM_VOICES_CONFIG["audio_subdir"] / voice_info["audio_filename"]
    if audio_path.exists():
        audio_path.unlink()
    
    # 删除文本文件
    text_path = user_dir / USER_CUSTOM_VOICES_CONFIG["texts_subdir"] / voice_info["text_filename"]
    if text_path.exists():
        text_path.unlink()
    
    # 更新元数据
    del metadata["voices"][voice_id]
    metadata["updated_time"] = datetime.now().isoformat()
    save_user_metadata(username, metadata)
    
    logger.info(f"用户 {username} 删除自定义录音: {voice_info['voice_name']} (ID: {voice_id})")
    
    return {
        "message": "自定义录音删除成功",
        "username": username,
        "voice_id": voice_id,
        "voice_name": voice_info["voice_name"]
    }

def generate_speech_with_custom_voice_logic(
    text: str,
    username: str,
    voice_id: str,
    gender: Optional[str] = None,
    pitch: Optional[str] = None,
    speed: Optional[str] = None
) -> TTSResponse:
    """使用用户自定义录音生成语音的核心逻辑"""
    logger = logging.getLogger(__name__)
    
    username = validate_username_format(username)
    user_dir = get_user_custom_dir(username)
    
    if not user_dir.exists():
        raise ValueError("用户不存在")
    
    metadata = get_user_metadata(username)
    
    if voice_id not in metadata.get("voices", {}):
        raise ValueError("自定义录音不存在")
    
    voice_info = metadata["voices"][voice_id]
    audio_path = user_dir / USER_CUSTOM_VOICES_CONFIG["audio_subdir"] / voice_info["audio_filename"]
    
    if not audio_path.exists():
        raise ValueError("音频文件不存在")
    
    # 使用现有的TTS逻辑生成语音
    result = generate_speech_with_prompt_logic(
        text=text,
        prompt_text=voice_info["text"],
        gender=gender,
        pitch=pitch,
        speed=speed,
        prompt_audio_path=str(audio_path)
    )
    
    # 添加自定义录音信息
    result.character_used = f"{username}:{voice_info['voice_name']}"
    
    logger.info(f"使用用户 {username} 的自定义录音 {voice_info['voice_name']} 生成语音成功")
    
    return result

def generate_speech_with_character_logic(request: TTSWithCharacterRequest):
    """使用角色库生成语音的核心逻辑"""
    logger = logging.getLogger(__name__)
    
    # 验证参数
    validate_parameters(request.gender, request.pitch, request.speed)
    
    # 加载角色库
    characters = load_voice_characters()
    if request.character_id not in characters:
        raise ValueError(f"角色 {request.character_id} 不存在")
    
    character = characters[request.character_id]
    
    # 获取模型实例
    model = initialize_model()
    
    # 生成唯一ID和文件名
    audio_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{audio_id}_{request.character_id}.wav"
    save_path = os.path.join(model_config['save_dir'], filename)
    
    logger.info(f"使用角色 {character.name} 生成语音: {request.text[:50]}...")
    
    # 执行推理
    try:
        with torch.no_grad():
            wav = model.inference(
                request.text,
                prompt_speech_path=character.audio_file,
                prompt_text=character.prompt_text,
                gender=request.gender or character.gender,
                pitch=request.pitch,
                speed=request.speed,
            )
            sf.write(save_path, wav, samplerate=16000)

            del wav

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise e

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        gc.collect()
    
    # 更新使用时间
    update_model_usage()
    
    logger.info(f"Audio generated successfully: {save_path}")
    
    return TTSResponse(
        message="Speech generated successfully with character",
        audio_id=audio_id,
        audio_path=save_path,
        timestamp=datetime.now().isoformat(),
        character_used=character.name
    )

    
def generate_speech_with_prompt_logic(
    text: str,
    prompt_text: Optional[str] = None,
    gender: Optional[str] = None,
    pitch: Optional[str] = None,
    speed: Optional[str] = None,
    prompt_audio_path: Optional[str] = None,
    target_duration: Optional[float] = None 
):
    """生成语音的核心逻辑（支持音频prompt）"""
    logger = logging.getLogger(__name__)
    
    validate_parameters(gender, pitch, speed)
    model = initialize_model()
    
    audio_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{audio_id}.wav"
    save_path = os.path.join(model_config['save_dir'], filename)
    
    logger.info(f"Generating speech with prompt for text: {text[:50]}...")
    
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech_path=prompt_audio_path,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch,
            speed=speed,
        )
        
    if target_duration and target_duration > 0:
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
   
        
    max_val = np.max(np.abs(wav))
   
    if max_val > 0:
        wav = wav / max_val * 0.95
        logger.info(f"音频归一化完成: 原始峰值 {max_val:.4f} -> 放大至 0.95")
    
    if target_duration and target_duration > 0:
        wav = adjust_audio_duration(wav, target_duration, sr=16000)
    sf.write(save_path, wav, samplerate=16000)
    
    # 更新使用时间
    update_model_usage()
    
    logger.info(f"Audio generated successfully: {save_path}")
    
    return TTSResponse(
        message="Speech generated successfully with prompt",
        audio_id=audio_id,
        audio_path=save_path,
        timestamp=datetime.now().isoformat()
    )

class TimelineDialogueLine(BaseModel):
    """时间轴对话行模型"""
    start: str
    end: str  
    role: str
    text: str
    voice_id: str
    speed: float = 1.0

class TimelineDialogueRequest(BaseModel):
    """时间轴对话请求"""
    dialogue_lines: List[TimelineDialogueLine]

class TimelineDialogueResponse(BaseModel):
    """时间轴对话响应"""
    audio_files: List[Dict[str, str]]
    timestamp: str

def parse_timestamp(timestamp: str) -> float:
    """解析时间戳为秒数"""
    parts = timestamp.split(':')
    hours = float(parts[0])
    minutes = float(parts[1]) 
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def generate_timeline_dialogue_logic(request: TimelineDialogueRequest):
    """时间轴对话生成核心逻辑"""
    characters = load_voice_characters()
    audio_files = []
    
    for line in request.dialogue_lines:
        # 判断voice_id类型：包含冒号则为用户自定义，否则为角色库
        if ':' in line.voice_id:
            # 用户自定义录音格式 "username:voice_id"
            username, voice_id = line.voice_id.split(':', 1)
            result = generate_speech_with_custom_voice_logic(
                text=line.text,
                username=username,
                voice_id=voice_id,
                speed="low" if line.speed < 1.0 else ("high" if line.speed > 1.0 else "moderate")
            )
        else:
            # 角色库格式
            if line.voice_id not in characters:
                raise ValueError(f"角色 {line.voice_id} 不存在")
            
            result = generate_speech_with_character_logic(
                TTSWithCharacterRequest(
                    text=line.text,
                    character_id=line.voice_id,
                    speed="low" if line.speed < 1.0 else ("high" if line.speed > 1.0 else "moderate")
                )
            )
        
        start_time = parse_timestamp(line.start)
        end_time = parse_timestamp(line.end)
        
        audio_files.append({
            "role": line.role,
            "text": line.text,
            "audio_path": result.audio_path,
            "audio_id": result.audio_id,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "voice_id": line.voice_id
        })
    
    return TimelineDialogueResponse(
        audio_files=audio_files,
        timestamp=datetime.now().isoformat()
    )
    
 
def adjust_audio_duration(wav_data: np.ndarray, target_duration: float, sr: int = 16000) -> np.ndarray:
    """
    【高保真对齐】绝不使用 time_stretch 变速，只做裁剪(Trim)和填充(Pad)。
    确保 100% 保留模型生成的原始音质。
    """
    logger = logging.getLogger(__name__)
    if target_duration is None or target_duration <= 0.1:
        return wav_data

    current_samples = len(wav_data)
    target_samples = int(target_duration * sr)
    
    # 1. 生成过长：切除静音 -> 还是长则截断 (保音质优先)
    if current_samples > target_samples:
        wav_trimmed, _ = librosa.effects.trim(wav_data, top_db=30)
        trimmed_len = len(wav_trimmed)
        
        if trimmed_len <= target_samples:
            # 切除静音后能塞进去，补齐剩余部分
            pad_width = target_samples - trimmed_len
            return np.pad(wav_trimmed, (0, pad_width), mode='constant')
        else:
            # 严重过长，执行硬截断并淡出
            # logger.warning(f"音频过长 ({current_samples/sr:.2f}s > {target_duration:.2f}s)，执行截断。")
            wav_trimmed = wav_trimmed[:target_samples]
            fade_len = int(0.05 * sr)
            if len(wav_trimmed) > fade_len:
                wav_trimmed[-fade_len:] *= np.linspace(1, 0, fade_len)
            return wav_trimmed

    # 2. 生成过短：补静音 (完美音质)
    elif current_samples < target_samples:
        pad_width = target_samples - current_samples
        return np.pad(wav_data, (0, pad_width), mode='constant')

    return wav_data

def _get_prompt_candidates(subtitles: List[SmartSubItem]) -> List[Dict]:
    """
    清洗并筛选最佳参考片段
    """
    candidates = []
    for item in subtitles:
        duration_sec = (item.end_time - item.start_time) / 1000.0
        text_len = len(item.text.strip())
        
        if duration_sec <= 0.5: continue 
        
        cps = text_len / duration_sec
        score = 0
        
        # 评分逻辑
        if cps < 1.0: score -= 50    # 可能是背景音
        elif cps > 9.0: score -= 50  # 可能是ASR错误
        else: score += 10
        
        if 3.0 <= duration_sec <= 8.0: score += 20
        elif 2.0 <= duration_sec < 3.0: score += 10
        elif duration_sec > 10.0: score -= 10
            
        candidates.append({
            "start": item.start_time,
            "end": item.end_time,
            "text": item.text, # 绑定原始文本
            "score": score,
            "duration": duration_sec
        })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def _merge_smart_subtitles(subtitles: List[SmartSubItem], threshold_ms: float) -> List[Dict]:
    """
    智能聚合字幕，返回原始列表供后续筛选
    """
    if not subtitles: return []
    sorted_subs = sorted(subtitles, key=lambda x: x.start_time)
    units = []
    current_group = [sorted_subs[0]]
    
    def pack_group(group):
        return {
            "speaker": group[0].speaker,
            "text_merged": "，".join([s.new_text for s in group]),
            "start_ms": group[0].start_time,
            "end_ms": group[-1].end_time,
            "sub_list": group # 保留原始列表
        }
    
    for i in range(1, len(sorted_subs)):
        prev = current_group[-1]
        curr = sorted_subs[i]
        gap = curr.start_time - prev.end_time
        if curr.speaker == prev.speaker and gap <= threshold_ms:
            current_group.append(curr)
        else:
            units.append(pack_group(current_group))
            current_group = [curr]
            
    if current_group:
        units.append(pack_group(current_group))
    return units

def _run_inference(model, text, prompt_path, prompt_text, speed):
    """封装推理调用"""
    with torch.no_grad():
        wav = model.inference(
            text=text,
            prompt_speech_path=prompt_path,
            prompt_text=prompt_text,
            speed=speed
        )
    if hasattr(wav, 'cpu'):
        wav = wav.cpu().numpy()
    return wav



def generate_aligned_speech(model, merged_unit: MergedUnit, candidate: Dict, full_audio: np.ndarray, sr: int): 
    """ 
    自适应语速生成函数 (生产环境版)
    1. 计算语速并生成音频
    2. 自动去除生成结果的静音
    3. 如果净时长严重超时，自动调整语速重试
    4. 最后进行物理对齐
    """ 
    logger = logging.getLogger(__name__) 

    # --- 1. 基础计算 --- 
    target_dur = (merged_unit.end_time_ms - merged_unit.start_time_ms) / 1000.0
    target_text_len = len(merged_unit.text_merged) 
    
    prompt_dur = candidate["duration"] 
    prompt_text_len = len(candidate["text"]) 
    
    # 动态计算 Prompt 语速 (防止除0或短文本偏差)
    if prompt_dur < 0.5 or prompt_text_len < 2: 
        prompt_cps = 3.5
    else: 
        prompt_cps = prompt_text_len / prompt_dur

    req_cps = target_text_len / target_dur
    
    # 计算倍率
    base_speed = req_cps / prompt_cps
    init_speed = max(0.8, min(base_speed, 1.5)) 

    # --- 2. 截取 Prompt 音频 --- 
    p_start = int((candidate["start"] / 1000.0) * sr) 
    p_end = int((candidate["end"] / 1000.0) * sr) 
    
    # 边界检查
    p_end = min(p_end, len(full_audio)) 
    prompt_wav = full_audio[p_start:p_end] 
    
    # 去除 Prompt 静音 (这对提取音色很重要)
    prompt_wav_trimmed, _ = librosa.effects.trim(prompt_wav, top_db=25) 
    
    # 创建临时文件供推理使用
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_prompt: 
        sf.write(tmp_prompt.name, prompt_wav_trimmed, sr) 
        tmp_prompt_path = tmp_prompt.name

    try: 
        # --- 3. 第一轮生成 ---
        wav_1_raw = _run_inference(model, merged_unit.text_merged, tmp_prompt_path, candidate["text"], init_speed) 
        
        # 立即清洗生成结果的静音
        wav_1, _ = librosa.effects.trim(wav_1_raw, top_db=30) 
        dur_1 = len(wav_1) / sr
        
        final_wav = wav_1
        
        # --- 4. 误差校验与重试 ---
        # 只有净时长依然超时 (>15%) 才重试
        if dur_1 > target_dur * 1.15: 
            correction = dur_1 / target_dur
            new_speed = min(init_speed * correction * 1.05, 1.6) 
            
            # logger.info(f"时长修正重试: Speed {init_speed:.2f} -> {new_speed:.2f}") 
            
            wav_2_raw = _run_inference(model, merged_unit.text_merged, tmp_prompt_path, candidate["text"], new_speed) 
            wav_2, _ = librosa.effects.trim(wav_2_raw, top_db=30) 
            final_wav = wav_2

        # --- 5. 物理对齐 (Padding/Trimming) ---
        return adjust_audio_duration(final_wav, target_dur, sr) 

    finally: 
        if os.path.exists(tmp_prompt_path): 
            os.unlink(tmp_prompt_path) 

def process_smart_dubbing(request: SmartDubbingRequest): 
    logger = logging.getLogger(__name__) 
    
    # --- 1. 检查输入文件 --- 
    if not os.path.exists(request.original_audio_path): 
        raise FileNotFoundError(f"原声文件不存在: {request.original_audio_path}") 
        
    model = initialize_model() 
    
    # --- 2. 加载音频并获取时长 --- 
    full_audio, sr = librosa.load(request.original_audio_path, sr=16000, mono=True) 
    audio_duration_sec = librosa.get_duration(y=full_audio, sr=sr) 
    
    # --- 3. 计算总时长 (防止截断) --- 
    last_subtitle_end_ms = 0
    if request.subtitles: 
        last_subtitle_end_ms = max(s.end_time for s in request.subtitles) 
    
    last_subtitle_end_sec = last_subtitle_end_ms / 1000.0
    final_duration_sec = max(audio_duration_sec, last_subtitle_end_sec) 
    
    # 初始化画布
    final_track = np.zeros(int(final_duration_sec * sr) + 16000) 
    
    # --- 4. 聚合字幕 ---
    merged_units_data = _merge_smart_subtitles(request.subtitles, request.merge_threshold_ms) 
    logger.info(f"智能聚合完成: {len(request.subtitles)} -> {len(merged_units_data)} 段") 
    
    # --- 5. 循环处理 ---
    for i, unit_data in enumerate(merged_units_data): 
        # 筛选参考音频
        candidates = _get_prompt_candidates(unit_data['sub_list']) 
        if not candidates: 
            # logger.warning(f"片段 {i} 无有效参考音频，跳过") 
            continue
        
        # 构造对象
        temp_unit = MergedUnit( 
            speaker=unit_data['speaker'], 
            text_merged=unit_data['text_merged'], 
            prompt_text="", 
            start_time_ms=unit_data['start_ms'], 
            end_time_ms=unit_data['end_ms'], 
            best_prompt_start=0, best_prompt_end=0
        ) 
        
        # 生成音频
        aligned_wav = generate_aligned_speech( 
            model=model, 
            merged_unit=temp_unit, 
            candidate=candidates[0], 
            full_audio=full_audio, 
            sr=sr
        ) 
        
        # 计算拼接位置 (绝对定位)
        start_idx = int((unit_data['start_ms'] / 1000.0) * sr) 
        end_idx = start_idx + len(aligned_wav) 
        
        # 动态扩容 (双重保险) 
        if end_idx > len(final_track): 
            new_track = np.zeros(end_idx + 16000) 
            new_track[:len(final_track)] = final_track
            final_track = new_track
            
        # 淡入淡出处理
        fade_len = int(0.05 * sr) 
        if len(aligned_wav) > fade_len * 2: 
            aligned_wav[:fade_len] *= np.linspace(0, 1, fade_len) 
            aligned_wav[-fade_len:] *= np.linspace(1, 0, fade_len) 
            
        # 叠加到总轨道
        final_track[start_idx:end_idx] += aligned_wav

    # --- 6. 导出 --- 
    output_filename = request.output_filename or f"smart_dub_{uuid.uuid4().hex[:8]}.wav" 
    save_path = os.path.join(model_config['save_dir'], output_filename) 
    
    # 归一化防止爆音
    max_val = np.max(np.abs(final_track)) 
    if max_val > 0.95: 
        final_track = final_track / max_val * 0.95
        
    sf.write(save_path, final_track, sr) 
    logger.info(f"配音生成完成: {save_path}") 
    return save_path






