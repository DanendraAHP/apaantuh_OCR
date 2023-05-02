import yaml
MODEL_DIR = 'models'
#config for API
with open("configs/mjsynth.yml", "r") as stream:
    try:
        tf_model_config = yaml.safe_load(stream)['dataset_builder']
    except yaml.YAMLError as exc:
        print(exc)

TEXT_DETECTION_CONFIG = {
    'east_path' : f'{MODEL_DIR}/frozen_east_text_detection.pb',
    'min_confidence' : 0.5,
    'east_width' : 320,
    'east_height': 320,
}

OCR_CONFIG = {
    'model_type' : 'CRNN', #CRNN or pytesseract,
    #for pytesseract config
    'tesseract_config' : ("-l ind --oem 1 --psm 8"),
    #for CRNN config
    'tf_model_path' : f'{MODEL_DIR}/SavedModel/',
    'tf_model_config' : tf_model_config
}