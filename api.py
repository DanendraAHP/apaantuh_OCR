
#############
##true file##
#############

from flask import Flask,request,jsonify
import os
from werkzeug.utils import secure_filename
from src.utils import decode_img
from src.inference_models import OCRModel, TextDetectionModel
from src.config import OCR_CONFIG

PATH = os.getcwd()
IMAGE_LOCATION = os.path.join(PATH,"images")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/', methods=['POST'])
def upload():
    respond = []
    if request.method == 'POST':
        if not request.json.get("image") or request.json.get("image") is None:
            respond.append({
                "message":"image not found"
            })
            return jsonify(respond)
        if not request.json.get("filename") or request.json.get("filename") is None :
            respond.append({
                "message":"filename not found"
            })
            return jsonify(respond)
        filename = request.json.get("filename")
        imgstring = request.json.get("image")
        if imgstring and allowed_file(filename):
            filename = secure_filename(filename)
            img = decode_img(imgstring)
            if OCR_CONFIG['model_type']=='CRNN':
                text = ocr_model.ocr_with_crnn(img)
            else:
                text = ocr_model.ocr_with_tesseract(img)
            respond.append({
                "filename":filename,
                "text":text,
            })
        else:
            respond.append({
            "message":"type file not permitted"
            })
    else:
        respond.append({
            "message":"request error"
        })
    return jsonify(respond)

if __name__ == "__main__":
    #init model
    text_detection_model = TextDetectionModel()
    ocr_model = OCRModel(text_detection_model)
    app.run(threaded=True, port=5000)
    #app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 443))