# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

import os, argparse
import requests

def translate_to_spanish(sentence:str) -> str:

	url = 'http://localhost:5000/translate'

	myobj = {'text': sentence, 
			 'source_vectorization': 'models/english_vectorization.pkl', 
			 'target_vectorization': 'models/spanish_vectorization.pkl', 
			 'model': 'models/translator_transformer.keras'}

	message = requests.post(url, json = myobj).json()

	return message['translated']

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='English to Spanish Translator')
	parser.add_argument('input_sentence', type=str, help='English sentence')
	args = parser.parse_args()

	print(' '.join(translate_to_spanish(args.input_sentence).split()[1:-1]))

