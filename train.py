"""
    Created by Ismail Bachchar
"""

#imports
import utils
import model

import pandas as pd
import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

tf.get_logger().setLevel('INFO')

# Configure Tensorflow to use the GPU
tf.config.experimental.set_visible_devices([], 'GPU')

#global vars
# IMAGES_PATH = "./data/flickr30k_images"
IMAGES_PATH = "./data2/Images"


#model parameters
# DenseNet201, Xception, VGG16, ResNet50, InceptionV3, InceptionResNetV2

# cnn_base_model = 'densenet201'
# embedding_size = 256
# before_last_layer_size = 128
# training_idx = "4k_images_1" #using flickr8k dataset
# learning_rate = 1e-3
# batch_size = 64

cnn_base_model = 'vgg16'
embedding_size = 256
before_last_layer_size = 128
training_idx = "2k_images_1" #using flickr8k dataset
learning_rate = 1e-3
batch_size = 64

# cnn_base_model = 'densenet201'
# embedding_size = 415
# before_last_layer_size = 256
# training_idx = "2"
# learning_rate = 1e-3
# batch_size = 64

# cnn_base_model = 'densenet201'
# embedding_size = 626
# before_last_layer_size = 256
# training_idx = "3"
# learning_rate = 1e-4
# batch_size = 64



# #read raw data
# data = pd.read_csv("./data/captions.csv", index_col=0)
# #read preprocessed data
# train = pd.read_csv("./data/train_set.csv", index_col=0)
# val = pd.read_csv("./data/val_set.csv", index_col=0)
# print("Read data", f"train set: {train.shape}, validation set:{val.shape}")

# #read 30k data
# data = pd.read_csv("./data2/captions.txt")
# data = utils.clean_text(data)
# #read preprocessed data
# train = pd.read_csv("./data2/train_set.csv", index_col=0)
# val = pd.read_csv("./data2/val_set.csv", index_col=0)
# print("Read data", f"train set: {train.shape}, validation set:{val.shape}")

# #read 4k data from flickr8k
# data = pd.read_csv("./data2/captions_4k.csv")
# data = utils.clean_text(data)
# #read preprocessed data
# train = pd.read_csv("./data2/train_set_4k.csv", index_col=0)
# val = pd.read_csv("./data2/val_set_4k.csv", index_col=0)
# print("Read data ... \n", f"train set: {train.shape}, validation set:{val.shape}")

#read 2k data from flickr8k
data = pd.read_csv("./data2/captions_1k.csv", index_col=0) #already pre-processed
# data = utils.clean_text(data)

#read preprocessed data
train = pd.read_csv("./data2/train_set_1k.csv", index_col=0)
val = pd.read_csv("./data2/val_set_1k.csv", index_col=0)
print("Read data ... \n", f"train set: {train.shape}, validation set:{val.shape}")



#create the tokenizer and fit it to the whole dataset captions
captions = data['caption'].tolist()
del data

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)
print(f"Done making the tokenizer with the vocabulary size: {vocab_size}, and maximum length: {max_length}")


#create the CNN encoder model first
print("creating the CNN encoder ...")
encoderCNN = model.CNN(cnn_base_model=cnn_base_model, embedding_size=embedding_size, 
                       data=train, images_folder=IMAGES_PATH)



#create the final model combining the CNN encoder and LSTM
modell = model.create_model(encoderCNN, embedding_size, max_length, vocab_size,
                before_last_layer_size)



#compile the model
modell.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy())


#data generator for train and validation sets
train_generator = utils.DataGenerator(df=train, X_col='image', y_col='caption', batch_size=batch_size, 
                                      tokenizer=tokenizer, vocab_size=vocab_size, images_folder=IMAGES_PATH, 
                                      max_length=max_length, encoderCNN=encoderCNN)

validation_generator = utils.DataGenerator(df=val, X_col='image', y_col='caption', batch_size=batch_size, 
                                           tokenizer=tokenizer, vocab_size=vocab_size, images_folder=IMAGES_PATH, 
                                           max_length=max_length, encoderCNN=encoderCNN)



checkpoint_path = f"./results/training_{training_idx}/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

#model training
history = modell.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        callbacks=[cp_callback])

#save the entire model
modell.save(f'./results/training_{training_idx}/model')

#save history
with open(f"./results/training_{training_idx}/history.pkl", "wb") as f:
    pickle.dump(history.history, f)
    f.close()