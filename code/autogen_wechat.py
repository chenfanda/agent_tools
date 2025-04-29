from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import httpx
import logging
import json
import sys
import os
from pathlib import Path
import time


async def wechat_send(name:str,message: str) -> str:
    """Send information by wechat"""
    processed_params= {"contact_name": name,
                    "message": message}
    async with httpx.AsyncClient() as client:
        tool_response= await client.post(
        "http://localhost:8003/tools/wechat/search_and_send",
        json=processed_params,
        timeout=300.0
        )
    return tool_response.json()


model_client = OpenAIChatCompletionClient(
    model="qwen2.5-coder:7b-instruct-fp16",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[wechat_send],
    system_message="当对话中提到发送信息时,使用工具完成任务，微信联系人提取对话文中的人名。",
)
await Console(agent.run_stream(task='写一首李白的将进酒,写完后使用微信发给联系人：陈浩'))
