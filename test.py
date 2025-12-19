import requests
import re
import os
import json
from datetime import datetime

# ================= 配置区域 =================
# 后端接口地址
API_URL = "http://127.0.0.1:8010/smart_dubbing/run"

AUDIO_PATH = "./vocals.wav" 

# 2. ASR 原始字幕文件路径 (用于提取原词和说话人)
ASR_PATH = "./asr_original.srt"

# 3. 新台词字幕文件路径 (用于生成语音)
NEW_SUB_PATH = "./funning.srt"

# ===========================================

def parse_srt_time_to_ms(time_str):
    """将 00:00:01,460 格式转换为毫秒 (float)"""
    try:
        # 统一处理逗号或点号
        time_str = time_str.replace(',', '.')
        hours, minutes, seconds = time_str.split(':')
        seconds = float(seconds)
        total_ms = (int(hours) * 3600 + int(minutes) * 60 + seconds) * 1000
        return total_ms
    except Exception as e:
        print(f"时间解析错误: {time_str}")
        return 0.0

def parse_srt(file_path):
    """简单的 SRT 解析器"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则匹配 SRT 块
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})\n((?:.|\n)*?)(?=\n\n|\Z)', re.MULTILINE)
    matches = pattern.findall(content)
    
    parsed_data = []
    for match in matches:
        index, start_str, end_str, text_block = match
        # 清理文本中的换行符
        text = text_block.replace('\n', ' ').strip()
        
        parsed_data.append({
            "index": index,
            "start_ms": parse_srt_time_to_ms(start_str),
            "end_ms": parse_srt_time_to_ms(end_str),
            "text": text
        })
    return parsed_data

def extract_speaker_and_clean(text):
    """
    通用清洗函数：
    1. 提取 [Speaker N] ID
    2. 返回清洗掉标签后的纯文本
    """
    # 查找 Speaker ID
    match = re.search(r'\[Speaker (\d+)\]', text)
    speaker_id = "speaker_0" # 默认值
    if match:
        speaker_id = f"speaker_{match.group(1)}"
    
    # 彻底清除 [Speaker N] 标签，只保留人话
    clean_text = re.sub(r'\[Speaker \d+\]', '', text).strip()
    
    return speaker_id, clean_text

def main():
    print("1. 解析字幕文件...")
    asr_subs = parse_srt(ASR_PATH)
    new_subs = parse_srt(NEW_SUB_PATH)

    if len(asr_subs) != len(new_subs):
        print(f"警告: ASR字幕行数 ({len(asr_subs)}) 与新台词行数 ({len(new_subs)}) 不一致！")
        print("将以较短的为准进行合并...")

    # 2. 组装请求数据 (SmartSubItem 结构)
    smart_subtitles = []
    
    min_len = min(len(asr_subs), len(new_subs))
    
    for i in range(min_len):
        asr_item = asr_subs[i]
        new_item = new_subs[i]
        
        # --- 关键修复开始 ---
        
        # 1. 处理原始字幕：提取 Speaker ID 和 纯文本
        speaker_id, clean_orig_text = extract_speaker_and_clean(asr_item['text'])
        
        # 2. 处理新台词：同样需要清洗 [Speaker X] 标签！
        # 我们不需要新台词里的 speaker_id (通常沿用原始的)，但必须把文本洗干净
        _, clean_new_text = extract_speaker_and_clean(new_item['text'])
        
        # --- 关键修复结束 ---
        
        # 构造数据项
        item = {
            "id": str(asr_item['index']),
            "start_time": asr_item['start_ms'], # 毫秒
            "end_time": asr_item['end_ms'],     # 毫秒
            "text": clean_orig_text,            # 原文 (Prompt Text) - 已清洗
            "new_text": clean_new_text,         # 新文 (Input Text) - 已清洗 [Speaker] 标签
            "speaker": speaker_id               # 说话人
        }
        smart_subtitles.append(item)

    # 3. 构造完整请求体
    payload = {
        "subtitles": smart_subtitles,
        "original_audio_path": AUDIO_PATH,
        "output_filename": f"validation_test_{datetime.now().strftime('%H%M%S')}.wav",
        "merge_threshold_ms": 500.0 # 聚合阈值
    }

    print(f"2. 准备发送请求，包含 {len(smart_subtitles)} 条字幕数据...")
    print(f"   原声路径: {AUDIO_PATH}")
    
    try:
        # 发送 POST 请求
        response = requests.post(API_URL, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 验证成功！")
            print(f"生成音频路径: {result.get('audio_path')}")
            print(f"音频 ID: {result.get('audio_id')}")
        else:
            print("\n❌ 请求失败")
            print(f"状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 连接失败: 请检查后端服务是否已启动")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()
