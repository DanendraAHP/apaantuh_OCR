import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import DATASET_CONFIG

#load mnist dataset
def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    return x_train, y_train, x_test, y_test

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
    #combining dataset
    mnist_data_train, mnist_label_train, mnist_data_test, mnist_label_test = load_mnist_dataset()
    az_data, az_labels = load_az_dataset(az_filepath)
    az_labels+=10
    az_data_train, az_data_test, az_label_train, az_label_test = train_test_split(az_data, az_labels, test_size=0.2, random_state=42, stratify=az_labels)
    img_train = np.vstack((mnist_data_train, az_data_train))
    img_test = np.vstack((mnist_data_test, az_data_test))
    label_train = np.hstack((mnist_label_train, az_label_train))
    label_test = np.hstack((mnist_label_test, az_label_test))
    #resize
    num_classess = len(set(label_train))
    label_test = tf.one_hot(label_test, num_classess)
    label_train = tf.one_hot(label_train, num_classess)
    img_train = img_train.reshape((img_train.shape[0], img_train.shape[1], img_train.shape[2], 1))
    img_test = img_test.reshape((img_test.shape[0], img_test.shape[1], img_test.shape[2], 1))
    return img_train, label_train, img_test, label_test, num_classess

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
    #keras layer for preprocessing and augmentation
    self.resize_and_rescale = tf.keras.Sequential([
      tf.keras.layers.Resizing(self.img_height, self.img_width),
      tf.keras.layers.Rescaling(scale=self.scale, offset=self.offset)
    ])
    self.data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.RandomRotation(0.2),
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
  def create_tf_dataset(self, x_train, y_train, x_test, y_test):
    #create tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    #preprocess and augment data
    train_dataset = self.preprocess_augment(train_dataset, augment=True)
    test_dataset = self.preprocess_augment(test_dataset)
    return train_dataset, test_dataset

  def prepare_dataset(self):
    x_train, y_train, x_test, y_test, num_classess = combining_dataset(self.az_filepath)
    self.train_dataset, self.test_dataset = self.create_tf_dataset(x_train, y_train, x_test, y_test)
    self.num_classess = num_classess