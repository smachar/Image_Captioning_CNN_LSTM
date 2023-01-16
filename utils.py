"""
    Created by Ismail Bachchar
"""

#imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.pyplot as plt

import tensorflow.keras.preprocessing.image as tfImage
import tensorflow.keras.utils as tfUtils
import tensorflow.keras.preprocessing as tfPreprocessor



def read_image(path, img_size=224):
    """read, resize and normalize an image saved in the path

    Args:
        path (str): the path where the image is saved
        img_size (int): the new image size. Defaults to 299

    Returns:
        array: the image normalized by 255
    """
    img = tfImage.load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = tfImage.img_to_array(img)
    
    return img/255.

def display_images(df, images_folder, nsamples=5, figsize=(15, 15)):
    """to display a set of images

    Args:
        df (pd.DataFrame): dataframe of image paths in the 'image' column and their corresponding 
        captions in the 'caption' column
        figsize (tuple of ints): figure size for the display
        nsamples (int): number of images to display from the df
    """
    df.reset_index(drop=True, inplace=True)
    plt.figure(figsize = figsize)
    n = 0
    for i in range(nsamples):
        n+=1
        plt.subplot(5 , 5, n)
        plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
        image = read_image(os.path.join(images_folder, df.image[i]))
        plt.imshow(image)
        plt.title("\n".join(wrap(df.caption[i], 20)))
        plt.axis("off")
        
def clean_text(data):
    """preprocess and clean text captions by:
    1) converting everythin to lowercase
    2) remove special chars, digits and empty (extra) space
    3) add <start> and <end> token to the start and end of every caption

    Args:
        data (pd.DataFrame): df with the 'caption' column to be cleaned

    Returns:
        pd.DataFrame: df with 'caption' column cleaned
    """
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]",""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+"," "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word)>1]))
    data['caption'] = "<start> "+data['caption']+" <end>"
    
    return data

def plot_model_performance(history):
    """plot the model's training and validation loss throughout the epochs

    Args:
        history (tensorflow.keras.callbacks.History): the training's saved history
    """
    plt.figure(figsize=(20,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def token_to_word(idx, tokenizer):
    """map the token to its correspinding word

    Args:
        idx (int): the token's index
        tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): the captions' tokenizer

    Returns:
        str or None: returns the word if found otherwise a None is returned
    """
    
    for word, index in tokenizer.word_index.items():
        if index==idx:
            return word
    return None

class DataGenerator(tfUtils.Sequence):
    """custom data generator based on the tf Sequence generator class

    Args:
        tensorflow.keras.utils.Sequence: Tensorflow Sequence generator
    """
    
    def __init__(self, df, tokenizer, vocab_size, max_length, encoderCNN, images_folder,
                 X_col='image', y_col='caption', batch_size=64, shuffle=True):
        """initialize the generator

        Args:
            df (pd.DataFrame): the data dataframe
            tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): the captions' tokenizer
            vocab_size (int): the vocabulary size after the tokenization of all data
            max_length (int): maximum caption's length that is presented in the data
            encoderCNN (object): the object of CNN pre-trained encoder class
            images_folder (str): the images folder path
            X_col (str, optional): input columns. Defaults to 'image'.
            y_col (str, optional): output column. Defaults to 'caption'.
            batch_size (int, optional): number of examples per batch. Defaults to 64.
            shuffle (bool, optional): either to shuffle or not on every epoch. Defaults to True.
        """
    
        #dataframe with 'image' and 'caption' columns
        self.df = df.copy()
        #input column, default set to the 'image' column
        self.X_col = X_col
        #output column, default set to the 'caption' column
        self.y_col = y_col
        #number of examples per batch, default set to 64
        self.batch_size = batch_size
        #the text tokenizer fitted on the captions data
        self.tokenizer = tokenizer
        #vocabulary size that was created by the tokenizer
        self.vocab_size = vocab_size
        #maximum sequence/sentece length between all the captions in the data
        self.max_length = max_length
        #the CNN encoder that extracts image's features
        self.encoderCNN = encoderCNN
        #either to shuffle or not on every epoch
        self.shuffle = shuffle
        #data length
        self.n = len(self.df)
        self.images_folder = images_folder
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __getitem__(self, index):
        """
        return batch samples as X_1 storing the images' features, X_2 storing the comulative tokens,
            and y storing the actual next to predict token (one-hot encoded)
        """
    
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        X1, X2, y = self.__get_data(batch)        
        return (X1, X2), y
    
    def __get_data(self, batch):
        
        #to store the batch data
        X1, X2, y = list(), list(), list()
        
        images = batch[self.X_col].tolist()
           
        for image in images:
            #get image's features extracted from the CNN encoder
            feature = self.encoderCNN.predict_sample(image=image, images_folder=self.images_folder)[0]
            #get image's captions from the data
            captions = batch.loc[batch[self.X_col]==image, self.y_col].tolist()
            for caption in captions:
                #tokenize the caption
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                #loop through the caption's tokens appending every one at a time
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    #append padding tokens to the sequence to extend its length to match the max_length
                    in_seq = tfPreprocessor.sequence.pad_sequences([in_seq], maxlen=self.max_length)[0]
                    #onehot encoding of the sequence's tokens into the vocab_size classes
                    out_seq = tfUtils.to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    #append the image's features to X1
                    X1.append(feature)
                    #append the cumulative tokens to X2 
                    X2.append(in_seq)
                    #append the actual next-to-predict token in the sequence
                    y.append(out_seq)
            
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                
        return X1, X2, y
    