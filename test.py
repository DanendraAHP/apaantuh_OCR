from src.utils import img_to_byte
import json
import requests
#img_path = "D:/Kuliah/S2/PPT/OCR/example_images/indomaret.jpg"
url = 'http://192.168.1.103:443/predict/'
img_path = "D:/Kuliah/S2/PPT/OCR/example_images/indomaret.jpg"
data = img_to_byte(img_path)
data = json.loads(data)
x = requests.post(url, json = data)
print(x.text)