import tensorflow as tf
from src.config import MODEL_CONFIG, DATASET_CONFIG

class OCRModel:
    def __init__(self):
        self.model_type = MODEL_CONFIG['MODEL_TYPE']
        self.fine_tune_at = MODEL_CONFIG['FINE_TUNE_AT']
        self.learning_rate = MODEL_CONFIG['LEARNING_RATE']
        #set up for pre-trained model
        if self.model_type!='baseline':
            self.base_model = MODEL_CONFIG['BASE_MODEL'][self.model_type]
            self.preprocess_input = MODEL_CONFIG['PREPROCESS_LAYER'][self.model_type]
            self.base_model.trainable = True
            for layer in self.base_model.layers[:self.fine_tune_at]:
                layer.trainable = False
    def create_model(self):
        if self.model_type=='baseline':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(36, activation='softmax')
            ])
        else :
            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            prediction_layer = tf.keras.layers.Dense(DATASET_CONFIG['NUM_CLASS'])
            inputs = tf.keras.Input(shape=(DATASET_CONFIG['IMG_WIDTH'], DATASET_CONFIG['IMG_HEIGHT'], DATASET_CONFIG['NUM_CHANNEL']))
            x = self.preprocess_input(inputs)
            x = self.base_model(x)
            x = global_average_layer(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = prediction_layer(x)
            self.model = tf.keras.Model(inputs, outputs)

    def train(self, train_ds, val_ds):
        self.create_model()
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', 
            verbose=1,
            patience=MODEL_CONFIG['ES_PATIENCE']
        )  
        mc=tf.keras.callbacks.ModelCheckpoint(
            filepath = MODEL_CONFIG['MODEL_PATH'], 
            monitor='val_loss', 
            mode='min', 
            save_best_only=True,
            verbose=1,
            save_weights_only=True
        )  
        self.history = self.model.fit(train_ds,
            epochs=MODEL_CONFIG['EPOCHS'],
            validation_data=val_ds,
            callbacks=[es, mc]
        )
