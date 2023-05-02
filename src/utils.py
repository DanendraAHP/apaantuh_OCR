import re
import base64
from PIL import Image
import numpy as np
from io import BytesIO
import json

def crop_img(image, box):
    startX, startY, endX, endY = box
    return image[startY:endY, startX:endX,:]
def post_process_text(text):
    text = text.lower()
    pattern = re.compile('[\W_]+')
    text = pattern.sub('', text)
    return text
def img_to_byte(img_path):
    with open(img_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read())
    data = {"filename": img_path, "image": img_str.decode()}
    return json.dumps(data)
def decode_img(img_str):
    img_str = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_str)).convert("RGB")
    return np.array(img)

# def preprocess_img(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     return img