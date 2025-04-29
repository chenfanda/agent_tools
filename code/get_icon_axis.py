import pyautogui
import base64
import io
import requests

def get_screenshot():
    """get screenshot """
    img = pyautogui.screenshot()
    return img

def image_mark(img_base64,url=None):
    """use model OmniParser-v2.0 to get screenshot element bbox axis"""
    if not url:
        url = 'http://127.0.0.1:6688/parse/'
    files={"base64_image":img_base64}
    response = requests.post(url, json=files)
    output=response.json()
    return output
    
def covert_base64(image):
    """transform image to base64 """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG") 
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64
    
def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def decode_image(base64_image,image_path):
    image_data = base64.b64decode(base64_image)
    with open(image_path, "wb") as image_file:
        image_file.write(image_data)

def convert_bbox_to_mouse_coords(bbox_list):
    """
    Convert a list of dictionaries containing bbox coordinates to mouse coordinates.
    
    Args:
        bbox_list (list): List of dictionaries with bbox coordinates in format
                          [x_min, y_min, x_max, y_max] as normalized values (0-1)
    
    Returns:
        list: List of dictionaries with the original data plus added mouse coordinates
    """
    # Get screen resolution
    screen_width, screen_height = pyautogui.size()
    
    # Process each item in the list
    result = []
    for item in bbox_list:
        if 'bbox' in item:
            # Extract bbox coordinates
            x_min, y_min, x_max, y_max = item['bbox']
            
            # Convert normalized coordinates to pixel values
            pixel_x_min = int(x_min * screen_width)
            pixel_y_min = int(y_min * screen_height)
            pixel_x_max = int(x_max * screen_width)
            pixel_y_max = int(y_max * screen_height)
            
            # Calculate center point for mouse coordinates
            current_mouse_x = (pixel_x_min + pixel_x_max) // 2
            current_mouse_y = (pixel_y_min + pixel_y_max) // 2
            
            # Add mouse coordinates to the item
            item_with_coords = item.copy()
            item_with_coords['currentMouseX'] = current_mouse_x
            item_with_coords['currentMouseY'] = current_mouse_y
            
            result.append(item_with_coords)
        else:
            # If item doesn't have bbox, just add it unchanged
            result.append(item)
    [key.pop(i) for key in result for i in ['bbox','source'] ]
    
    # df=pd.DataFrame(result)[['type','interactivity','content','currentMouseX','currentMouseY']]
    return result



def get_image_info():
    """获取截屏图片并且转化为base64格式"""
    img=get_screenshot()
    img_str=covert_base64(img)
    return img_str


def get_screen_info():
    """获取屏幕元素坐标信息"""
    img_str=get_image_info()
    bbox_info=image_mark(img_str)
    result=convert_bbox_to_mouse_coords(bbox_info['parsed_content_list'])
    return result

def save_screenshot(image_path):
    """保存截屏图片到本地目录"""
    img_str=get_image_info()
    decode_image(img_str,image_path)
    return f'image save in {image_path}'
