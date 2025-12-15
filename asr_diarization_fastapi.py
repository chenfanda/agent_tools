from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os,sys
import shutil
from pathlib import Path

# 导入修改后的模块
from asr_diarization import whisper_with_diarization, run_whisper_asr, create_simple_srt
from audio_refine import extract_audio_from_video, separate_audio_tracks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = "storage"
os.makedirs(DATA_ROOT, exist_ok=True)

class CORSStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

app.mount("/static", CORSStaticFiles(directory=DATA_ROOT), name="static")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    project_id: str = Form(...),
    enable_diarization: str = Form("true"),
    enable_vocal_separation: str = Form("true")
):
    try:
        project_dir = Path(DATA_ROOT) / user_id / project_id
        source_dir = project_dir / "source"
        processed_dir = project_dir / "processed"
        
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        input_filename = file.filename
        input_path = source_dir / input_filename
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_ext = input_path.suffix.lower()
        is_video = file_ext in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        audio_working_path = input_path

        print(f"DEBUG: 接收到的 enable_diarization 参数: '{enable_diarization}'")
        
        should_diarize = str(enable_diarization).lower() == 'true'
        should_separate = str(enable_vocal_separation).lower() == 'true'
        
        print(f"DEBUG: 转换后的开关状态 -> 角色分离: {should_diarize}, 人声提取: {should_separate}") 
        # 1. 如果是视频，提取音频
        if is_video:
            audio_working_path = processed_dir / "extracted_audio.wav"
            print(f"Extracting audio to: {audio_working_path}")
            extract_audio_from_video(str(input_path), str(audio_working_path))

        vocals_url = ""
        backing_url = ""
        final_asrc_audio = str(audio_working_path)

        # 2. 人声分离 (Demucs)
        if should_separate:
            print("Starting vocal separation...")
            separate_audio_tracks(str(audio_working_path), str(processed_dir))
            
            vocals_path = processed_dir / "vocals.wav"
            backing_path = processed_dir / "accompaniment.wav"
            
            if vocals_path.exists():
                vocals_url = f"/static/{user_id}/{project_id}/processed/vocals.wav"
                # 如果分离成功，ASR 应当使用纯人声，效果更好
                final_asrc_audio = str(vocals_path)
            if backing_path.exists():
                backing_url = f"/static/{user_id}/{project_id}/processed/accompaniment.wav"
        
        subtitle_filename = f"{input_path.stem}.srt"
        subtitle_path = processed_dir / subtitle_filename
        
        segments = []
        
        # 3. ASR 转录 (Faster-Whisper) + 说话人分离 (NeMo)
        print(f"Starting ASR on: {final_asrc_audio}")

        if should_diarize:
            segments = whisper_with_diarization(
                audio_input=final_asrc_audio,
                whisper_model="large-v3", # 这里的名称主要用于日志，实际路径在 asr_diarization.py 内部定死了
                output_srt=str(subtitle_path)
            )
        else:
            # 仅 ASR
            whisper_result = run_whisper_asr(final_asrc_audio, "large-v3")
            create_simple_srt(whisper_result, str(subtitle_path))
            segments = whisper_result['segments']

        with open(subtitle_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        return JSONResponse(content={
            "success": True,
            "data": {
                "subtitles": {
                    "format": "srt",
                    "content": srt_content,
                    "url": f"/static/{user_id}/{project_id}/processed/{subtitle_filename}"
                },
                "source_resources": {
                    "video": f"/static/{user_id}/{project_id}/source/{input_filename}",
                    "audioVocals": vocals_url,
                    "audioBacking": backing_url
                },
                "metadata": {
                    "segments_count": len(segments)
                }
            }
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动 API
    uvicorn.run(app, host="0.0.0.0", port=8008)
