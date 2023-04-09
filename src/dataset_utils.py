import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from src.config import DATASET_CONFIG
import time
#load mnist dataset
def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    data = np.vstack([x_train, x_test])
    labels = np.hstack([y_train, y_test])
    return data, labels

#load az dataset
def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	data = []
	labels = []
	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))
		# update the list of data and labels
		data.append(image)
		labels.append(label)
	return np.array(data, dtype='float'), np.array(labels, dtype='int')

#combining data for training models
def combining_dataset(az_filepath):
    combine_start = time.time()
    #combining dataset
    #load mnist data
    start = time.time()
    mnist_data, mnist_label = load_mnist_dataset()
    end = time.time()
    print(f'[INFO] load mnist data took {end-start} second')
    #load az data
    start = time.time()
    az_data, az_labels = load_az_dataset(az_filepath)
    end = time.time()
    print(f'[INFO] load az data took {end-start} second')
    az_labels+=10
    data = np.vstack([az_data, mnist_data])
    labels = np.hstack([az_labels, mnist_label])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
    #resize and one hot
    num_classess = DATASET_CONFIG['NUM_CLASS']
    y_train = tf.one_hot(y_train, num_classess)
    y_test = tf.one_hot(y_test, num_classess)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    combine_end = time.time()
    print(f'[INFO] combining dataset took {combine_end-combine_start} seconds')
    return X_train, X_test, y_train, y_test

class Dataset:
  def __init__(self):
    #constant
    self.autotune = tf.data.AUTOTUNE
    self.img_height = DATASET_CONFIG['IMG_HEIGHT']
    self.img_width = DATASET_CONFIG['IMG_WIDTH']
    self.batch_size = DATASET_CONFIG['DATA_BATCH_SIZE']
    self.az_filepath = DATASET_CONFIG['AZ_FILEPATH']
    self.scale = DATASET_CONFIG['SCALE']
    self.offset = DATASET_CONFIG['OFFSET']
    self.num_classess = DATASET_CONFIG['NUM_CLASS']
    self.kfold_split = DATASET_CONFIG['KFOLD_SPLIT']
    self.augment = DATASET_CONFIG['AUGMENT_DATA']
    #keras layer for preprocessing and augmentation
    self.resize_and_rescale = tf.keras.Sequential([
      tf.keras.layers.Resizing(self.img_height, self.img_width),
      tf.keras.layers.Rescaling(scale=self.scale, offset=self.offset)
    ])
    self.data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomHeight(0.1),
      tf.keras.layers.RandomRotation(0.1),
      tf.keras.layers.RandomWidth(0.1)
    ])
  def preprocess(self, img, label):
    img = self.resize_and_rescale(img)
    img = tf.image.grayscale_to_rgb(img)
    return img,label
  def preprocess_augment(self, ds, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(self.preprocess)
    # Batch all datasets.
    ds = ds.batch(self.batch_size)
    # Use data augmentation only on the training set.
    if augment:
      ds = ds.map(lambda x, y: (self.data_augmentation(x, training=True), y), 
                  num_parallel_calls=self.autotune)
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=self.autotune)

  #creating tf dataset for training model
  def create_tf_dataset(self, x_train, y_train, x_val, y_val):
    #create tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    #preprocess and augment data
    train_dataset = self.preprocess_augment(train_dataset, self.augment)
    val_dataset = self.preprocess_augment(val_dataset)
    return train_dataset, val_dataset
  #get combined data and get index from KFOLD
  def build(self):
    start = time.time()
    X_train, X_test, y_train, y_test = combining_dataset(self.az_filepath)
    # kf = KFold(n_splits=self.kfold_split, shuffle=True, random_state=42)
    # self.kfold_train_idx = []
    # self.kfold_val_idx = []
    # for i, (train_index, val_index) in enumerate(kf.split(self.data)):
    #   self.kfold_train_idx.append(train_index)
    #   self.kfold_val_idx.append(val_index)
    self.train_dataset, self.val_dataset = self.create_tf_dataset(
        x_train=X_train, 
        y_train=y_train,
        x_val = X_test,  
        y_val=y_test
    )
    end = time.time()
    print(f'[INFO] building dataset tooks {end-start} seconds') 