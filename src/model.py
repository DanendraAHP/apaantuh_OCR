import tensorflow as tf
from src.config import MODEL_CONFIG, DATASET_CONFIG
import matplotlib.pyplot as plt
import time

class OCRModel:
    def __init__(self, kfold_split):
        self.model_type = MODEL_CONFIG['MODEL_TYPE']
        self.fine_tune_at = MODEL_CONFIG['FINE_TUNE_AT']
        self.learning_rate = MODEL_CONFIG['LEARNING_RATE']
        self.model_folder = MODEL_CONFIG['MODEL_FOLDER']
        self.kfold_split = kfold_split
        self.model_folder = f'{self.model_folder}_{self.kfold_split}'
        #set up for pre-trained model
        if self.model_type!='baseline':
            self.base_model = MODEL_CONFIG['BASE_MODEL'][self.model_type]
            self.preprocess_input = MODEL_CONFIG['PREPROCESS_LAYER'][self.model_type]
            self.base_model.trainable = True
            for layer in self.base_model.layers[:self.fine_tune_at]:
                layer.trainable = False
    def create_model(self):
        start = time.time()
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
            prediction_layer = tf.keras.layers.Dense(DATASET_CONFIG['NUM_CLASS'], activation='softmax')
            inputs = tf.keras.Input(shape=(DATASET_CONFIG['IMG_WIDTH'], DATASET_CONFIG['IMG_HEIGHT'], DATASET_CONFIG['NUM_CHANNEL']))
            x = self.preprocess_input(inputs)
            x = self.base_model(x)
            x = global_average_layer(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = prediction_layer(x)
            self.model = tf.keras.Model(inputs, outputs)
        end = time.time()
        print(f'[INFO] initializing model took {end-start} seconds')
    def train(self, train_ds, val_ds):
        start = time.time()
        self.create_model()
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy', mode='max', 
            verbose=1,
            patience=MODEL_CONFIG['ES_PATIENCE']
        )  
        mc=tf.keras.callbacks.ModelCheckpoint(
            filepath = f'{self.model_folder}/cp.ckpt', 
            monitor='val_categorical_accuracy', 
            mode='max', 
            save_best_only=True,
            verbose=1,
            save_weights_only=True
        )  
        self.history = self.model.fit(train_ds,
            epochs=MODEL_CONFIG['EPOCHS'],
            validation_data=val_ds,
            callbacks=[es, mc]
        )
        end = time.time()
        print(f'[INFO] training model took {end-start} seconds')
    def load(self, model_type, model_path):
        start = time.time()
        self.model_type = model_type
        self.create_model()
        self.model.load_weights(model_path)
        end = time.time()
        print(f'[INFO] loading the model took {end-start} seconds')
    def eval_model(self, val_data):
        print('evaluating the model'.center(20,'-'))
        loss, accuracy = self.model.evaluate(val_data)
        return loss, accuracy
    def visualize_history(self):
        # summarize history for accuracy
        plt.plot(self.history.history['categorical_accuracy'])
        plt.plot(self.history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig(f'{self.model_folder}/categorical_accuracy.png')
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig(f'{self.model_folder}/loss.png')