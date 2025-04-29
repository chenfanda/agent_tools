from mcp.server.fastmcp import FastMCP
import sys
import uvicorn
from typing import List, Dict, Any, Optional
import asyncio
import logging

# Import the browser context and handler
sys.path.append('/oper/ch/code')
from browser_use.dom.service import DomService
from browser_use.dom.views import DOMElementNode, SelectorMap
from browser_use.utils import time_execution_sync

# Create a browser context instance (you'll need to adjust this according to your setup)
from browser_context import BrowserContext  # Import your browser context
browser_context = BrowserContext()  # Initialize your browser context
browser_handler = None  # This will be initialized in startup

# Initialize MCP
mcp = FastMCP("browser tools")

# Setup logging
logger = logging.getLogger(__name__)

@mcp.on_startup
async def startup():
    global browser_handler
    # Assuming you have a BaseHandler class implementation from base.py
    from tools.handlers.base import BaseHandler
    browser_handler = BaseHandler(browser_context)
    logger.info("Browser handler initialized")

@mcp.on_shutdown
async def shutdown():
    if browser_handler:
        await browser_handler.cleanup()
    logger.info("Browser handler cleaned up")

# Basic Operations

@mcp.tool()
async def go_to_url(url: str) -> Dict[str, Any]:
    """导航到指定URL"""
    return await browser_handler.go_to_url({"url": url})

@mcp.tool()
async def click_element(index: int) -> Dict[str, Any]:
    """点击指定索引的元素"""
    return await browser_handler.click_element({"index": index})

@mcp.tool()
async def input_text(index: int, text: str) -> Dict[str, Any]:
    """在指定元素中输入文本"""
    return await browser_handler.input_text({"index": index, "text": text})

@mcp.tool()
async def extract_content(goal: str) -> Dict[str, Any]:
    """提取页面内容"""
    return await browser_handler.extract_content({"goal": goal})

@mcp.tool()
async def scroll_page(direction: str = "down", amount: str = "medium") -> Dict[str, Any]:
    """
    滚动页面
    参数:
    - direction: 滚动方向，可选 "up" 或 "down"
    - amount: 滚动量，可选 "small", "medium", "large", "page"
    """
    return await browser_handler.scroll({"direction": direction, "amount": amount})

@mcp.tool()
async def wait(time_seconds: float = 2.0, selector: Optional[str] = None) -> Dict[str, Any]:
    """
    等待指定时间或元素出现
    参数:
    - time_seconds: 等待的秒数
    - selector: 可选的CSS选择器，如果指定则等待该元素出现
    """
    params = {"time": time_seconds}
    if selector:
        params["selector"] = selector
    return await browser_handler.wait(params)

# Tab Operations

@mcp.tool()
async def get_tabs() -> Dict[str, Any]:
    """获取所有标签页信息"""
    return await browser_handler.get_tabs({})

@mcp.tool()
async def create_tab(url: str = "about:blank") -> Dict[str, Any]:
    """创建新标签页并导航到指定URL"""
    return await browser_handler.create_tab({"url": url})

@mcp.tool()
async def switch_tab(tab_id: str) -> Dict[str, Any]:
    """切换到指定ID的标签页"""
    return await browser_handler.switch_tab({"tab_id": tab_id})

@mcp.tool()
async def close_tab() -> Dict[str, Any]:
    """关闭当前标签页"""
    return await browser_handler.close_tab({})

# Element Finding Operations

@mcp.tool()
async def highlight_elements(viewport_expansion: int = 500) -> Dict[str, Any]:
    """高亮并获取页面上的可点击元素"""
    return await browser_handler.highlight_elements({"viewport_expansion": viewport_expansion})

@mcp.tool()
async def find_element_by_text(text: str, partial_match: bool = True) -> Dict[str, Any]:
    """
    通过文本内容查找元素
    参数:
    - text: 要查找的文本
    - partial_match: 是否部分匹配（默认为True）
    """
    return await browser_handler.find_element_by_text({
        "text": text,
        "partial_match": partial_match,
        "highlight_elements": True
    })

@mcp.tool()
async def find_element_by_attribute(attribute: str, value: str, partial_match: bool = True) -> Dict[str, Any]:
    """
    通过属性查找元素
    参数:
    - attribute: 属性名
    - value: 属性值
    - partial_match: 是否部分匹配（默认为True）
    """
    return await browser_handler.find_element_by_attribute({
        "attribute": attribute,
        "value": value,
        "partial_match": partial_match,
        "highlight_elements": True
    })

# Advanced Operations

@mcp.tool()
async def inject_script(script: str) -> Dict[str, Any]:
    """注入并执行JavaScript脚本"""
    return await browser_handler.inject_script({"script": script})

# Combined Operations

@mcp.tool()
async def get_or_create_tab_with_url(url: str) -> Dict[str, Any]:
    """
    检查是否有包含指定URL的标签页，有则切换，无则创建新标签页
    参数:
    - url: 目标URL
    """
    return await browser_handler.get_or_create_tab_with_url({"url": url})

@mcp.tool()
async def find_and_click_by_text(text: str, partial_match: bool = True) -> Dict[str, Any]:
    """
    查找并点击包含指定文本的元素
    参数:
    - text: 要查找的文本
    - partial_match: 是否部分匹配（默认为True）
    """
    return await browser_handler.find_and_click_element_by_text({
        "text": text,
        "partial_match": partial_match
    })

@mcp.tool()
async def create_masked_tab(target_url: str) -> Dict[str, Any]:
    """
    创建一个带有数据遮罩的标签页
    参数:
    - target_url: 要导航到的URL
    """
    return await browser_handler.create_mask_interceptor({"target_url": target_url})

# General query handler
@mcp.tool()
async def process_browser_action(action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行通用浏览器操作
    参数:
    - action: 操作名称
    - parameters: 操作参数
    """
    params = {"action": action, **parameters}
    return await browser_handler.process_query(params)

if __name__ == "__main__":
    uvicorn.run(mcp, host="0.0.0.0", port=8800)
