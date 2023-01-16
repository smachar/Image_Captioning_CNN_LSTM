"""
    Created by Ismail Bachchar
"""

#imports
import utils as utils
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu

from tensorflow.keras.applications import DenseNet201, Xception, VGG16, ResNet50, InceptionV3, InceptionResNetV2
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfLayers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.preprocessing as tfPreprocessor

class CNN():
    """The CNN encoder that extracts images features using a pre-trained model from the followings:
    DenseNet201, Xception, VGG16, ResNet50, InceptionV3, InceptionResNetV2
    """
    def __init__(self, embedding_size, data, images_folder, cnn_base_model='densenet201'):
        
        if cnn_base_model=='densenet201':
            self.base_model = DenseNet201(include_top=False, input_shape=(224, 224, 3), 
                                       weights="imagenet", pooling='max')
        elif cnn_base_model=='xception':
            self.base_model = Xception(include_top=False, input_shape=(299, 299, 3), 
                                       weights="imagenet", pooling='max')
        elif cnn_base_model=='vgg16':
            self.base_model = VGG16(include_top=False, input_shape=(224, 224, 3), 
                                       weights="imagenet", pooling='max')
        elif cnn_base_model=='resnet50':
            self.base_model = ResNet50(include_top=False, input_shape=(299, 299, 3), 
                                       weights="imagenet", pooling='max')
        elif cnn_base_model=='inceptionv3':
            self.base_model = InceptionV3(include_top=False, input_shape=(299, 299, 3), 
                                       weights="imagenet", pooling='max')
        elif cnn_base_model=='inceptionresnetv2':
            self.base_model = InceptionResNetV2(include_top=False, input_shape=(299, 299, 3), 
                                       weights="imagenet", pooling='max')
        
        self.pre_trained_model = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        self.input = tfLayers.Input(shape=(self.base_model.output.shape[1],)) #shape [None, int] None is the batch size
        self.dense = tfLayers.Dense(embedding_size, activation='relu')(self.input)
        self.dense_reshaped = tfLayers.Reshape((1, embedding_size), input_shape=(embedding_size,))(self.dense)
        
        #dictionary to store images' features extracted from the pre-trained model (key=> image, and value=> features)
        self.features = self.predict_data(data, images_folder)
        
    def predict_data(self, data, images_folder):
        """Generate the image features (encoding) of all images presented in the data

        Args:
            data (pd.DataFrame): the data dataframe with the 'image' columns havinf the images' names
            images_folder (str): the images folder path

        Returns:
            dictionary: a dictionary containing as key the images' names and their corresponding encoding/features
        """
        #eg. model.input shape TensorShape([None, 224, 224, 3])
        img_size = self.base_model.input.shape[2]
        #to store the encodings
        features = {}
        for image in tqdm(data['image'].unique().tolist()):
            #read image
            img = utils.read_image(os.path.join(images_folder, image), img_size=img_size)
            #reshape the image
            img = np.expand_dims(img, axis=0)
            #extract the image's features using the CNN pre-trained model
            features[image] = self.pre_trained_model.predict(img, verbose=0)
            
        return features
    
    def predict_sample(self, image, images_folder):
        """extract one image's features

        Args:
            image (str): the image name
            images_folder (str): the images folder path

        Returns:
            np.array: the image's extracted features/encdoding
        """
        img = utils.read_image(os.path.join(images_folder, image), img_size=self.base_model.input.shape[2])
        #reshape the image
        img = np.expand_dims(img, axis=0)
        
        #return image's features/encoding
        return self.pre_trained_model.predict(img, verbose=0)
            
def create_model(encoderCNN, embedding_size, max_length, vocab_size,
                before_last_layer_size):
    """Create the final model where it combines the pre-trained CNN model with a LSTM layers

    Args:
        encoderCNN (object): the object of CNN pre-trained encoder class
        embedding_size (int): the desired size of sequence's embeddings (every token is encoded into a vectore 
            of this size)
        max_length (int): maximum caption's length that is presented in the data
        vocab_size (int): the vocabulary size after the tokenization of all data
        before_last_layer_size (int): the size of the before last layer (that has a size of vocab_size)

    Returns:
        tensorflow.keras.Model: the tensorflow created model
    """
    
    #creating the RNN decoder part
    input = tfLayers.Input(shape=(max_length,))
    embedding = tfLayers.Embedding(input_dim=vocab_size, output_dim=embedding_size)(input)
    #concatenate image features layer with the embedding layer
    merged = tfLayers.concatenate([encoderCNN.dense_reshaped, embedding], axis=1)
    #append to LSTM layer (only using one-lstm layer in this architecture)
    lstm_cnn = tfLayers.LSTM(embedding_size)(merged)
    dropout = tfLayers.Dropout(0.5)(lstm_cnn)
    
    #adding lstm last layer with CNN encoder output layer (features) (model approach 2)
    lstm_cnn = tfLayers.add([dropout, encoderCNN.dense])
    
    lstm_cnn = tfLayers.Dense(before_last_layer_size, activation='relu')(lstm_cnn)
    lstm_cnn = tfLayers.Dropout(0.5)(lstm_cnn)
    output = tfLayers.Dense(vocab_size, activation='softmax')(lstm_cnn)

    model = Model(inputs=[encoderCNN.input, input], outputs=output)
    
    return model

def predict_sample(encoderCNN, model, image, images_folder, tokenizer, max_length):
    """Predict the caption of a given image

    Args:
        encoderCNN (object): the object of CNN pre-trained encoder class
        model (tensorflow.keras.Model): the final created model that combines CNN and RNN models
        image (str): the image name
        images_folder (str): the images folder path
        tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): the captions' tokenizer
        max_length (int): maximum caption's length that is presented in the data

    Returns:
        tuple: the predicted caption and list of tokens' predictions probabilities
    """
    
    #encode the image first
    feature = encoderCNN.predict_sample(image=image, images_folder=images_folder)
    #to store tokens' predicted probabilities
    tokens_predictions = []
    #start token
    caption = "<start>"
    #loop through the maximum sequence length
    for i in range(max_length):
        #get the tokens of the words in caption
        sequence = tokenizer.texts_to_sequences([caption])[0]
        #padd the sequence to the max_length
        sequence = tfPreprocessor.sequence.pad_sequences([sequence], maxlen=max_length)

        #predict the next toke in the sequence
        y_pred = model.predict([feature, sequence], verbose=0)
        tokens_predictions.append(y_pred)
        y_pred = np.argmax(y_pred)
        
        #get the token's corresponding word
        word = utils.token_to_word(y_pred, tokenizer)
        
        #break if the word is None
        if word is None:
            break
            raise("the toke {y_pred} was not found on the vocabulary!")
            return
        
        #concatenate the next predicted word to get the full caption at the end
        caption+= " " + word
        
        #stop prediction if the end token is predicted
        if word == '<end>':
            break

    return caption, tokens_predictions

def predict_batch(encoderCNN, model, data, images_folder, tokenizer, max_length):
    """predict the captions of a batch of images

    Args:
        encoderCNN (object): the object of CNN pre-trained encoder class
        model (tensorflow.keras.Model): the final created model that combines CNN and RNN models
        data (pd.DataFrame): the data dataframe with the 'image' columns havinf the images' names
        images_folder (str): the images folder path
        tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): the captions' tokenizer
        max_length (int): maximum caption's length that is presented in the data
    """
    #dataframe to be retuned
    result = pd.DataFrame(data['image'].unique(), columns=['image'])
    #to stode the predicted captions of every unique image
    predicted_captions = []
    
    #loop through unique images in the data
    for image in tqdm(data['image'].unique().tolist()):
        #predict the image's caption and append to the predicted_captions list
        caption, _ = predict_sample(encoderCNN, model, image, images_folder, tokenizer, max_length)
        predicted_captions.append(caption)
    
    #add the predicted captions to the dataframe and return it
    result['predicted_caption'] = predicted_captions
    
    return result
        
def evaluate_model(result, data):
    """evaluate the model's predictions by calculating the blue score between
        the predicted caption against all possible actual captions of the images

    Args:
        result (pd.DataFrame): data frame of 'image' and 'predicted_caption' columns
        data (pd.DataFrame): the dataframe where images in the 'result' dataframe are presented
            with thei actual captions

    Returns:
        pd.DataFrame: the same 'result' with an extra column; the calculated bleu score
            of every image
    """
    #to store the blue scores
    scores = []
    #loop through images in the 'result' dataframe
    for i in range(len(result)):
        #get image name
        image = result['image'].iloc[i]
        #get predicted caption
        predicted_caption = result['predicted_caption'].iloc[i].split()
        
        #buils the reference captions list
        actual_possible_captions = []
        #loop through all captions of the current image
        for caption in data[data['image']==image]['caption'].tolist():
            actual_possible_captions.append(caption.split())
        
        #append the calculated bleu-4 score of the current image predicted caption and its references
        scores.append(sentence_bleu(actual_possible_captions, predicted_caption, weights=(0, 0, 0, 1)))
    
    result['bleu_score'] = scores
    
    return result
        