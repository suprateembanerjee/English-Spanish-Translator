# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

import os, re, pickle, random, string, argparse
from typing import List, Dict, Tuple
import numpy as np

import tensorflow as tf 
from tensorflow import keras 
from keras import layers, losses, metrics, activations, optimizers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from transformer import PositionalEmbedding, TransformerEncoder, TransformerDecoder
from text_vectorizer import TextVectorizer


def load_data(filepath:str) -> List[Tuple[str, str]]:
	'''
	Loads data from a file containing spanish and english sentences separated by tab.
	
	args:
	- filepath: String containing path to data.
	
	returns:
	- List of tuples, each containing a pair of english and spanish sentences.
	'''

	with open(filepath) as f:
	    lines = f.read().split('\n')[:-1]

	text_pairs = []

	for line in lines:
	    english, spanish = line.split('\t')
	    spanish = '[start] ' + spanish + ' [end]'
	    text_pairs.append((english, spanish))

	return text_pairs

def split_data(text_pairs:List[Tuple[str, str]], 
				ratio:dict={'train':0.7, 'val':0.15, 'test':0.15}) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
	'''
	Splits data into train, validation and test.
	
	args:
	- text_pairs: List of tuples, each containing a pair of english and spanish sentences.
	- ratio: Split ratio.
	
	returns:
	- Three list of tuples (train, validation, test), each containing pairs of english and spanish sentences.
	'''

	if sum(ratio.values()) != 1:
		print('Invalid ratio!')
		return

	num_val_samples = int(ratio['val'] * len(text_pairs))
	num_test_samples = int(ratio['test'] * len(text_pairs))
	num_train_samples = len(text_pairs) - num_val_samples - num_test_samples

	train_pairs = text_pairs[:num_train_samples]
	val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
	test_pairs = text_pairs[num_train_samples + num_val_samples:]

	return train_pairs, val_pairs, test_pairs

def adapt_vectorization(text:str, 
						vocab_size:int=15000, 
						sequence_length:int=20, 
						standardize:str='lower_and_strip_punctuation') -> TextVectorizer:
	'''
	Adapts a text vectorizer on a given dataset.
	
	args:
	- text: List of sentences in one language.
	- vocab_size: Size of vocabulary (maximum number of words under consideration from the corpus).
	- sequence_length: Length of sequence.
	- standardize: Method of standardization. Custom standardizations can be added in TextVectorizer implementation.
	
	returns:
	- TextVectorizer object containing adapted vectorization on text.
	'''

	vectorization = TextVectorizer(max_tokens=vocab_size,
                                   output_mode='int',
                                   output_sequence_length=sequence_length,
                                   standardize=standardize)
	vectorization.adapt(text)

	return vectorization

def make_dataset(pairs:List[Tuple[str, str]],
				 batch_size:int, 
				 source_vectorization:TextVectorizer, 
				 target_vectorization:TextVectorizer) -> tf.data.Dataset:
	'''
	Creates a tensorflow dataset out of raw data.
	
	args:
	- pairs: List of tuples containing english and spanish sentences.
	- batch_size: Batch size of dataset.
	- source_vectorization: Text vectorization of source language.
	- target_vectorization: Text vectorization of target language.
	
	returns:
	- Tensorflow dataset containing shuffled, prefetched, batched items from raw data.
	'''

	def format_dataset(eng, spa):

		eng = source_vectorization(eng)
		spa = target_vectorization(spa)

		return {'english': eng, 'spanish':spa[:, :-1]}, spa[:, 1:]

	eng_texts, spa_texts = zip(*pairs)
	eng_texts = list(eng_texts)
	spa_texts = list(spa_texts)

	dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
	dataset = dataset.batch(batch_size)
	dataset = dataset.map(format_dataset, num_parallel_calls=4)

	return dataset.shuffle(2048).prefetch(16).cache()

def get_model(sequence_length:int, 
			  vocab_size:int, 
			  embed_dim:int, 
			  dense_dim:int, 
			  num_heads:int) -> keras.Model:
	'''
	Creates a Transformer model out of specifications.
	
	args:
	- sequence_length: Length of sequences.
	- vocab_size: Size of vocabulary.
	- embed_dim: Positional Embedding dimensions.
	- dense_dim: Dense layer dimensions.
	- num_heads: Number of heads for transformer model.
	
	returns:
	- Keras model defining a Transformer architecture.
	'''

	encoder_inputs = keras.Input(shape=(None,), dtype=tf.int64, name='english')
	x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
	encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

	decoder_inputs = keras.Input(shape=(None,), dtype=tf.int64, name='spanish')
	x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
	x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
	x = layers.Dropout(0.5)(x)
	decoder_outputs = layers.Dense(vocab_size, activation=activations.softmax)(x)

	transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

	return transformer

def train(model:keras.Model, 
		  train_dataset:tf.data.Dataset, 
		  val_dataset:tf.data.Dataset, 
		  epochs:int, 
		  optimizer:KerasTensor, 
		  loss:KerasTensor, 
		  metrics:List[KerasTensor], 
		  callbacks:List[KerasTensor]) -> keras.callbacks.History:
	'''
	Trains a transformer model on a dataset.
	
	args:
	- model: Keras model defining a Transformer architecture.
	- train_dataset: Training dataset.
	- val_dataset: Validation dataset.
	- epochs: Number of epochs
	- optimizer: Choice of optimizer for training.
	- loss: Choice of loss function for training.
	- metrics: List of metrics to track during training.
	- callbacks: List of callbacks to track during training.

	
	returns:
	- Training history of the model, after training.
	'''

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	history = model.fit(train_dataset,
						epochs=epochs,
						validation_data=val_dataset,
						callbacks=callbacks)

	return history