import tensorflow as tf

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
}

MODEL_CONFIG = {
    'MODEL_TYPE' : 'mobilenet',#baseline/mobilenet/inception,
    'PREPROCESS_LAYER' : {
        'mobilenet' : tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    'BASE_MODEL' : {
        'mobilenet' : tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
    },
    'MODEL_FOLDER' : 'models/mobilenet',
    'FINE_TUNE_AT' : 50,
    'LEARNING_RATE' : 0.0001,
    'EPOCHS' : 1000,
    'ES_PATIENCE' : 5
}