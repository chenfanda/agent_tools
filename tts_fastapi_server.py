import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入工具函数和数据模型
from tts_utils import (
    setup_logging,
    model_unload_checker,
    load_voice_characters,
    initialize_model,
    model_instance,
    model_last_used,
    model_config,
    IDLE_TIME_MINUTES,
    USER_CUSTOM_VOICES_CONFIG,
    TTSRequest,
    TTSWithCharacterRequest,
    VoiceCharacter,
    TTSResponse,
    SaveCustomVoiceRequest,
    UserCustomVoice,
    SaveCustomVoiceResponse,
    generate_speech_with_character_logic,
    TimelineDialogueLine,
    TimelineDialogueRequest,
    TimelineDialogueResponse,
    generate_timeline_dialogue_logic,
    generate_speech_with_prompt_logic,
    generate_speech_with_custom_voice_logic,
    save_custom_voice_logic,
    get_user_custom_voices_logic,
    delete_user_custom_voice_logic,
    get_custom_voices_statistics,
    preprocess_audio,
    validate_username_format,
    SmartDubbingRequest, 
    process_smart_dubbing
)

setup_logging()
import logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spark TTS API with Voice Characters and Custom Voices",
    description="Text-to-Speech API using Spark TTS model with voice character library and user custom voices",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化监控线程"""
    logger.info("Starting TTS API service with voice characters and custom voices...")
    
    # 确保用户自定义录音基础目录存在
    base_dir = Path(USER_CUSTOM_VOICES_CONFIG["base_dir"])
    base_dir.mkdir(exist_ok=True)
    
    # 启动模型卸载检查线程
    unload_thread = threading.Thread(target=model_unload_checker, daemon=True)
    unload_thread.start()
    logger.info(f"模型自动卸载功能已启用 (空闲{IDLE_TIME_MINUTES}分钟后卸载)")
    
    logger.info("TTS API service started successfully")

@app.get("/")
async def root():
    """根路径健康检查"""
    return {"message": "Spark TTS API with Voice Characters and Custom Voices is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        characters = load_voice_characters()
        # with model_lock:
        model_loaded = model_instance is not None
        last_used = model_last_used.isoformat() if model_last_used else None
        
        # 获取用户自定义录音统计信息
        custom_stats = get_custom_voices_statistics()
        
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "model_last_used": last_used,
            "characters_loaded": len(characters),
            "custom_voices_users": custom_stats["total_users"],
            "custom_voices_total": custom_stats["total_custom_voices"],
            "idle_timeout_minutes": IDLE_TIME_MINUTES,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# 预设角色相关接口
@app.get("/characters", response_model=list[VoiceCharacter])
async def get_voice_characters():
    """获取所有可用的语音角色"""
    try:
        characters = load_voice_characters()
        return list(characters.values())
    except Exception as e:
        logger.error(f"获取角色列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取角色列表失败: {str(e)}")

@app.get("/characters/{character_id}", response_model=VoiceCharacter)
async def get_character_info(character_id: str):
    """获取特定角色信息"""
    try:
        characters = load_voice_characters()
        if character_id not in characters:
            raise HTTPException(status_code=404, detail="角色不存在")
        return characters[character_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取角色信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取角色信息失败: {str(e)}")

# 用户自定义录音接口
@app.post("/save_custom_voice", response_model=SaveCustomVoiceResponse)
async def save_custom_voice(
    username: str = Form(...),
    voice_name: str = Form(...),
    text: str = Form(...),
    description: str = Form(None),
    audio_file: UploadFile = File(...)
):
    """保存用户自定义录音"""
    temp_dir = None
    
    try:
        # 验证请求数据
        request_data = SaveCustomVoiceRequest(
            username=username,
            voice_name=voice_name,
            text=text,
            description=description
        )
        
        # 验证音频文件
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            raise HTTPException(status_code=400, detail="不支持的音频格式，请上传 WAV、MP3、FLAC 或 M4A 文件")
        
        # 创建临时目录处理音频
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, audio_file.filename)
        
        # 保存上传的音频文件
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        logger.info(f"用户 {username} 上传音频文件: {audio_file.filename}")
        
        # 预处理音频（格式转换、质量检查等）
        try:
            processed_audio_path = preprocess_audio(temp_audio_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"音频处理失败: {str(e)}")
        
        # 调用业务逻辑保存自定义录音
        return save_custom_voice_logic(
            username=username,
            voice_name=voice_name,
            text=text,
            description=description,
            processed_audio_path=processed_audio_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存自定义录音失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/user_custom_voices/{username}", response_model=List[UserCustomVoice])
async def get_user_custom_voices(username: str):
    """获取指定用户的自定义录音列表"""
    try:
        return get_user_custom_voices_logic(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"获取用户 {username} 自定义录音失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取自定义录音失败: {str(e)}")

@app.delete("/user_custom_voices/{username}/{voice_id}")
async def delete_user_custom_voice(username: str, voice_id: str):
    """删除指定的用户自定义录音"""
    try:
        return delete_user_custom_voice_logic(username, voice_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"删除用户 {username} 自定义录音 {voice_id} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@app.post("/tts_with_custom_voice", response_model=TTSResponse)
async def generate_speech_with_custom_voice(
    text: str = Form(...),
    username: str = Form(...),
    voice_id: str = Form(...),
    gender: str = Form(None),
    pitch: str = Form(None),
    speed: str = Form(None)
):
    """使用用户自定义录音生成语音"""
    try:
        return generate_speech_with_custom_voice_logic(
            text=text,
            username=username,
            voice_id=voice_id,
            gender=gender,
            pitch=pitch,
            speed=speed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"使用自定义录音生成语音失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语音生成失败: {str(e)}")

# 原有的TTS接口保持不变
@app.post("/tts_with_character", response_model=TTSResponse)
async def generate_speech_with_character(request: TTSWithCharacterRequest):
    """使用角色库生成语音"""
    try:
        return generate_speech_with_character_logic(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS generation with character failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")



@app.post("/tts_with_prompt", response_model=TTSResponse)
async def generate_speech_with_prompt(
    text: str = Form(...),
    prompt_text: str = Form(None),
    gender: str = Form(None),
    pitch: str = Form(None),
    speed: str = Form(None),
    target_duration: float = Form(None),
    prompt_audio: UploadFile = File(None)
):
    """生成语音（支持上传音频作为prompt）"""
    prompt_audio_path = None
    temp_dir = None
    
    try:
        if prompt_audio:
            temp_dir = tempfile.mkdtemp()
            raw_audio_path = os.path.join(temp_dir, prompt_audio.filename)
            
            with open(raw_audio_path, "wb") as buffer:
                shutil.copyfileobj(prompt_audio.file, buffer)
            
            logger.info(f"Prompt audio saved to: {raw_audio_path}")
            try:
                prompt_audio_path = preprocess_audio(raw_audio_path)
                logger.info(f"Preprocessed prompt audio: {prompt_audio_path}")
            except Exception as e:
                logger.error(f"Audio preprocessing failed: {e}")
                # 如果预处理失败，回退到原始文件（虽然可能会崩，但好过直接报错）
                prompt_audio_path = raw_audio_path
        
        return generate_speech_with_prompt_logic(
            text=text,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch,
            speed=speed,
            prompt_audio_path=prompt_audio_path,
            target_duration=target_duration
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS generation with prompt failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/download/{audio_id}")
async def download_audio(audio_id: str):
    """下载生成的音频文件"""
    try:
        save_dir = Path(model_config['save_dir'])
        audio_files = list(save_dir.glob(f"*{audio_id}*.wav"))
        
        if not audio_files:
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        audio_file = audio_files[0]
        
        if not audio_file.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=str(audio_file),
            media_type="audio/wav",
            filename=audio_file.name,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/audio/{audio_id}")
async def delete_audio(audio_id: str):
    """删除指定的音频文件"""
    try:
        save_dir = Path(model_config['save_dir'])
        audio_files = list(save_dir.glob(f"*{audio_id}*.wav"))
        
        if not audio_files:
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        deleted_files = []
        for audio_file in audio_files:
            if audio_file.exists():
                audio_file.unlink()
                deleted_files.append(audio_file.name)
        
        return {
            "message": f"Deleted {len(deleted_files)} audio file(s)",
            "deleted_files": deleted_files,
            "audio_id": audio_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/audio/list")
async def list_audio_files():
    """列出所有生成的音频文件"""
    try:
        save_dir = Path(model_config['save_dir'])
        if not save_dir.exists():
            return {"audio_files": []}
        
        audio_files = []
        for file_path in save_dir.glob("*.wav"):
            stat = file_path.stat()
            audio_files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {
            "total_files": len(audio_files),
            "audio_files": sorted(audio_files, key=lambda x: x["created_time"], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"List files failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")


@app.post("/tts_timeline_dialogue", response_model=TimelineDialogueResponse)
async def generate_timeline_dialogue(request: TimelineDialogueRequest):
    """时间轴多角色对话配音接口"""
    try:
        return generate_timeline_dialogue_logic(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"时间轴对话生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")
 
 
 
@app.post("/smart_dubbing/run")
async def run_smart_dubbing_endpoint(request: SmartDubbingRequest):
    """
    智能全流程配音接口：聚合 -> 选优Prompt -> 生成 -> 对齐 -> 拼接
    """
    try:
        
        output_path = process_smart_dubbing(request)
        
        return {
            "status": "success",
            "audio_path": output_path,
            "audio_id": os.path.basename(output_path).replace(".wav", "") 
        }
    except Exception as e:
        logger.error(f"Smart dubbing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    MODEL_DIR = os.getenv("MODEL_DIR", "pretrained_models/Spark-TTS-0.5B")
    SAVE_DIR = os.getenv("SAVE_DIR", "api_results")
    DEVICE = int(os.getenv("DEVICE", "0"))
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8010"))
    
    model_config.update({
        "model_dir": MODEL_DIR,
        "save_dir": SAVE_DIR,
        "device": DEVICE
    })
    
    logger.info(f"Starting server with config: {model_config}")
    
    uvicorn.run(
        "tts_fastapi_server:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
