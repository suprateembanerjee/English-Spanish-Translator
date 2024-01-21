# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]
# Usage
# python src/inference.py --input_sentence "I have never heard him speak English." 
# --source_vectorization "models/english_vectorization.pkl" 
# --target_vectorization "models/spanish_vectorization.pkl" 
# --model "models/translator_transformer.keras"


import tensorflow as tf 
from tensorflow import keras 
from keras import layers, losses, metrics, activations
import re, pickle, random
import numpy as np
import argparse, logging

from transformer import PositionalEmbedding, TransformerEncoder, TransformerDecoder
from text_vectorizer import TextVectorizer

def load_vectorizer(filepath:str) ->TextVectorizer:
    '''
    Loads a vectorizer from a pickle file.

    Parameters:
    filepath: Path to pickle file containing TextVectorizer data.

    Return:
    TextVectorizer object loaded from pickle file.
    '''

    vectorization_data = pickle.load(open(filepath, 'rb'))
    vectorizer = TextVectorizer.from_config(vectorization_data['config'])
    vectorizer.set_weights(vectorization_data['weights'])

    return vectorizer

def load_model(filepath:str) -> keras.Model:
    '''
    Loads a keras model from a pickle file.

    Parameters:
    filepath: Path to pickle file containing model.

    Return:
    Keras Model object loaded from pickle file.
    '''

    transformer = keras.models.load_model(filepath)
    return transformer


def translate(input_sentence:str, source_vectorization:TextVectorizer, target_vectorization:TextVectorizer, model:keras.Model) -> str:
    '''
    Performs translation inference based on the provided model.

    Parameters:
    input_sentence: English sentence to be translated.
    source_vectorization: English Vectorization.
    target_vectorization: Spanish Vectorization.
    model: Model to perform inference with.

    Return:
    String containing translated sentence in Spanish.
    '''

    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    max_decoded_sentence_length = 20

    tokenized_input_sentence = source_vectorization([input_sentence])                                                       
    translated = '[start]'

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([translated])[:,:-1]
        next_token_predictions = model.predict([tokenized_input_sentence, tokenized_target_sentence], verbose=0)
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])                                                    
        sampled_token = spa_index_lookup[sampled_token_index]                                                               
        translated += ' ' + sampled_token

        if sampled_token == '[end]':
            break
    
    return translated


if __name__ == '__main__':

    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description='English to Spanish Translator')
    parser.add_argument('-i', '--input_sentence', type=str, help='English sentence')
    parser.add_argument('-sv', '--source_vectorization', type=str, help='Source Vectorization')
    parser.add_argument('-tv', '--target_vectorization', type=str, help='Target Vectorization')
    parser.add_argument('-m', '--model', type=str, help='Model')

    args = parser.parse_args()

    source_vectorization = load_vectorizer(args.source_vectorization)
    target_vectorization = load_vectorizer(args.target_vectorization)
    model = load_model(args.model)

    print(translate(args.input_sentence, source_vectorization, target_vectorization, model))

