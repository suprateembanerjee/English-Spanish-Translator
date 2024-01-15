# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

import tensorflow as tf
from tensorflow import keras 
from keras import layers
import string, re

@keras.utils.register_keras_serializable(package='custom_layers', name='TextVectorizer')
class TextVectorizer(layers.Layer):
    '''English - Spanish Text Vectorizer'''

    def __init__(self, max_tokens=None, output_mode='int', output_sequence_length=None, standardize='lower_and_strip_punctuation', vocabulary=None, config=None):
        super().__init__()
        if config:
            self.vectorization = layers.TextVectorization.from_config(config)

        else:
            self.max_tokens = max_tokens
            self.output_mode = output_mode
            self.output_sequence_length = output_sequence_length
            self.vocabulary = vocabulary
            if standardize == 'spanish':
                self.vectorization = layers.TextVectorization(max_tokens=self.max_tokens,
                                                              output_mode=self.output_mode,
                                                              output_sequence_length=self.output_sequence_length,
                                                              vocabulary=self.vocabulary,
                                                              standardize=self.spanish_standardize)
            else:
                self.vectorization = layers.TextVectorization(max_tokens=self.max_tokens,
                                                              output_mode=self.output_mode,
                                                              output_sequence_length=self.output_sequence_length,
                                                              vocabulary=self.vocabulary)


    def spanish_standardize(self, input_string, preserve=['[', ']'], add=['¿']) -> str:
        strip_chars = string.punctuation
        for item in add:
            strip_chars += item
        
        for item in preserve:
            strip_chars = strip_chars.replace(item, '')

        lowercase = tf.strings.lower(input_string)
        output = tf.strings.regex_replace(lowercase, f'[{re.escape(strip_chars)}]', '')

        return output
    
    def __call__(self, *args, **kwargs):
        return self.vectorization.__call__(*args, **kwargs)
    
    def get_config(self):
        return {key: value if not callable(value) else None for key, value in self.vectorization.get_config().items()}
    
    def from_config(config):
        return TextVectorizer(config=config)
    
    def set_weights(self, weights):
        self.vectorization.set_weights(weights)

    def adapt(self, dataset):
        self.vectorization.adapt(dataset)
    
    def get_vocabulary(self):
        return self.vectorization.get_vocabulary()