# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]
# Usage: python src/train.py --data data --model models

import os, re, pickle, random, string, argparse
from typing import List, Dict, Tuple
import numpy as np

import tensorflow as tf 
from tensorflow import keras 
from keras import layers, losses, metrics, activations, optimizers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from transformer import PositionalEmbedding, TransformerEncoder, TransformerDecoder
from text_vectorizer import TextVectorizer
from train_utils import load_data, split_data, adapt_vectorization, make_dataset, get_model, train


def main(data_path:str, 
		 model_path:str,
		 optimizer:KerasTensor,
		 loss:KerasTensor,
		 metrics:List[KerasTensor],
		 epochs:int,
		 save:bool):

	text_pairs = load_data(f'{data_path}/external/spa-eng/spa.txt')
	random.Random(5).shuffle(text_pairs)
	train_pairs, val_pairs, test_pairs = split_data(text_pairs)

	train_english_texts = [pair[0] for pair in train_pairs]
	train_spanish_texts = [pair[1] for pair in train_pairs]

	vocab_size = 15000
	sequence_length = 20

	if os.path.exists(f'{data_path}/processed/train_dataset') and os.path.exists(f'{data_path}/processed/val_dataset') and os.path.exists(f'{data_path}/processed/test_dataset'):
		train_dataset = tf.data.Dataset.load(f'{data_path}/processed/train_dataset')
		test_dataset = tf.data.Dataset.load(f'{data_path}/processed/test_dataset')
		val_dataset = tf.data.Dataset.load(f'{data_path}/processed/val_dataset')

	else:
		source_vectorization = adapt_vectorization(train_english_texts)
		target_vectorization = adapt_vectorization(train_spanish_texts, sequence_length=sequence_length+1, standardize='spanish')
		pickle.dump({'config': source_vectorization.get_config(),
	             	 'weights': source_vectorization.get_weights()}, open(f'{model_path}/english_vectorization.pkl', 'wb'))

		pickle.dump({'config': target_vectorization.get_config(),
             	 	 'weights': target_vectorization.get_weights()}, open(f'{model_path}/spanish_vectorization.pkl', 'wb'))

		batch_size = 64
		train_dataset = make_dataset(train_pairs, batch_size, source_vectorization, target_vectorization)
		val_dataset = make_dataset(val_pairs, batch_size, source_vectorization, target_vectorization)
		test_dataset = make_dataset(test_pairs, batch_size, source_vectorization, target_vectorization)
		train_dataset.save(f'{data_path}/processed/train_dataset')
		val_dataset.save(f'{data_path}/processed/val_dataset')
		test_dataset.save(f'{data_path}/processed/test_dataset')

	
	embed_dim=256
	dense_dim=2048
	num_heads=8

	model = get_model(sequence_length, vocab_size, embed_dim, dense_dim, num_heads)

	callbacks = [keras.callbacks.ModelCheckpoint(f'{model_path}/translator_transformer.keras', save_best_only=True)] if save else None

	train(model,
		  train_dataset,
		  val_dataset,
		  epochs=epochs,
		  optimizer=optimizer,
		  loss=loss,
		  metrics=metrics,
		  callbacks = callbacks)


if __name__=='__main__':

	parser = argparse.ArgumentParser(description='English to Spanish Translator')
	parser.add_argument('-d', '--data_path', type=str, help='File path to data directory')
	parser.add_argument('-m', '--model_path', type=str, help='File path to model directory')
	parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs to train')
	parser.add_argument('optimizer', nargs='?', default=optimizers.RMSprop())
	parser.add_argument('loss', nargs='?', default=losses.SparseCategoricalCrossentropy())
	parser.add_argument('metrics', nargs='?', default=[metrics.SparseCategoricalAccuracy()])
	parser.add_argument('-s', '--save', action='store_true')

	args = parser.parse_args()

	main(data_path=args.data_path, 
		model_path=args.model_path, 
		optimizer=args.optimizer, 
		loss=args.loss, 
		metrics=args.metrics, 
		epochs=args.epochs,
		save=args.save)


