import os

# ==========================================================
# 🚀 核心修复：设置 Hugging Face 国内镜像地址
# 必须放在 import faster_whisper 之前，或者在脚本最开始执行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ==========================================================

import shutil
from faster_whisper import download_model
from demucs.pretrained import get_model

# === 配置路径 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 子目录
FW_DIR = os.path.join(MODELS_DIR, "faster-whisper")
DEMUCS_DIR = os.path.join(MODELS_DIR, "demucs")
NEMO_DIR = os.path.join(MODELS_DIR, "nemo")

def setup_directories():
    print(f"📂 创建模型根目录: {MODELS_DIR}")
    os.makedirs(FW_DIR, exist_ok=True)
    os.makedirs(DEMUCS_DIR, exist_ok=True)
    os.makedirs(NEMO_DIR, exist_ok=True)

def download_faster_whisper():
    print("\n⬇️  正在下载 Faster-Whisper 模型 (large-v3)...")
    print("ℹ️  已启用镜像加速: https://hf-mirror.com")
    
    try:
        # download_model 会自动使用 HF_ENDPOINT 环境变量
        model_path = download_model("large-v3", output_dir=FW_DIR)
        print(f"✅ Faster-Whisper 模型已就绪: {model_path}")
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("💡 提示: 如果镜像也不行，请检查服务器是否有外网权限。")

def migrate_demucs():
    print("\n⬇️  正在处理 Demucs 模型...")
    # Demucs 使用的是 Facebook 的服务器，镜像站可能不覆盖
    # 如果 Demucs 也下载失败，可能需要配置 HTTP_PROXY
    os.environ["TORCH_HOME"] = DEMUCS_DIR
    try:
        get_model('htdemucs')
        print(f"✅ Demucs 模型已就绪: {DEMUCS_DIR}")
    except Exception as e:
        print(f"⚠️ Demucs 下载失败: {e}")
        print("Demucs 模型通常较小，如果下载失败，可能需要手动下载。")

def migrate_nemo():
    print("\n📦 正在迁移 NeMo 模型...")
    # 你的旧路径逻辑
    old_nemo_path = os.path.join(PROJECT_ROOT, "diar_sortformer_4spk-v1", "diar_sortformer_4spk-v1.nemo")
    target_path = os.path.join(NEMO_DIR, "diar_sortformer_4spk-v1.nemo")
    
    if os.path.exists(target_path):
        print("✅ NeMo 模型已存在。")
    elif os.path.exists(old_nemo_path):
        shutil.copy(old_nemo_path, target_path)
        print("✅ NeMo 模型已从旧目录复制。")
    else:
        print("⚠️ 未找到 NeMo 模型，请手动放入 models/nemo 目录。")

if __name__ == "__main__":
    setup_directories()
    download_faster_whisper()
    migrate_demucs()
    migrate_nemo()
    print("\n🎉 所有模型准备完毕！")
