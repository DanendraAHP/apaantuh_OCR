import tensorflow as tf
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
