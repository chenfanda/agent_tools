from mcp.server.fastmcp import FastMCP
import sys
import uvicorn
from typing import List,Dict
sys.path.append('/oper/ch/code')
from get_icon_axis import get_image_info,get_screen_info,encode_image,save_screenshot
# from mcp.server import Server
mcp = FastMCP("mcp tools")



# @mcp.resource("echo://{message}")
# def echo_resource(message: str) -> str:
#     """Echo a message as a resource"""
#     return f"Resource echo: {message}"

@mcp.tool()
def screenshot_element_axis() -> List[Dict]:
    """获取电脑屏幕元素坐标，元素坐标用于pyautogui鼠标或键盘操控"""
    screen_info=get_screen_info()
    return screen_info

@mcp.tool()
def image_base64() -> str:
    """获取电脑屏幕截屏，并且把图片转换为base64字符串 """
    img=get_image_info()
    return img

@mcp.tool()
def read_image_local(image_path:str) -> str:
    """读取本地图片,并且转换为base64字符串 """
    img=encode_image(image_path)
    return img

@mcp.tool()
def save_image_local(image_path:str) -> str:
    """保存截屏图片到本地目录 """
    log_info=save_screenshot(image_path)
    return log_info
# @mcp.prompt()
# def echo_prompt(message: str) -> str:
#     """Create an echo prompt"""
#     return f"Please process this message: {message}"




# routes = [Mount("/message/", app=sse.handle_post_message)]
if __name__ == "__main__":
   uvicorn.run(mcp,host="0.0.0.0",port=8800)


