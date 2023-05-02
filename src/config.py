import tensorflow as tf
import yaml

MODEL_DIR = 'models'

DATASET_CONFIG = {
    'DATA_BATCH_SIZE':128,
    'IMG_HEIGHT':32,
    'IMG_WIDTH':32,
    'NUM_CHANNEL' : 3,
    'AZ_FILEPATH': 'dataset/A_Z Handwritten Data.csv',
    'NUM_CLASS' : 36,
    'SCALE' : 1./255,
    'OFFSET' : 0.0,
    'KFOLD_SPLIT' : 5,
    'AUGMENT_DATA' : True
}

MODEL_CONFIG = {
    'MODEL_TYPE' : 'inception',#baseline/mobilenet/efficientnet,
    'PREPROCESS_LAYER' : {
        'mobilenet' : tf.keras.applications.mobilenet_v2.preprocess_input,
        'efficientnet' : tf.keras.applications.efficientnet_v2.preprocess_input,
        #cant be used because mnist only have 28x28 meanwhile the model need at least 150x150
        'inception' : tf.keras.applications.inception_v3.preprocess_input
    },
    'BASE_MODEL' : {
        'mobilenet' : tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet'),
        'efficientnet' : tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet'),
        #cant be used because mnist only have 28x28 meanwhile the model need at least 150x150
        'inception' : tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    },
    'MODEL_FOLDER' : f'{MODEL_DIR}/inception',
    'FINE_TUNE_AT' : 50,
    'LEARNING_RATE' : 0.0001,
    'EPOCHS' : 1000,
    'ES_PATIENCE' : 5
}
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