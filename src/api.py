# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

from flask import Flask
from flask_restful import Resource, Api, reqparse
from tensorflow import keras
import json

from inference import load_vectorizer, load_model, translate

model = None
source_vectorization = None
target_vectorization = None

app = Flask(__name__)
api = Api(app)


class Translate(Resource):

	def __init__(self):
		self.reqparse = reqparse.RequestParser()
		self.reqparse.add_argument('text', type=str, required=True, location='json', help='Input sentence!')
		self.reqparse.add_argument('source_vectorization', type=str, required=True, location='json', help='English vectorization!')
		self.reqparse.add_argument('target_vectorization', type=str, required=True, location='json', help='Spanish vectorization!')
		self.reqparse.add_argument('model', type=str, required=True, location='json', help='Model')

		super().__init__()

	def post(self):
		args = self.reqparse.parse_args()
		input_sentence = args['text']
		global model, source_vectorization, target_vectorization

		if not source_vectorization:
			source_vectorization = load_vectorizer(args['source_vectorization'])
		if not target_vectorization:
			target_vectorization = load_vectorizer(args['target_vectorization'])
		if not model:
			model = load_model(args['model'])

		translated = translate(input_sentence, source_vectorization, target_vectorization, model)
		return {'translated':translated}

api.add_resource(Translate, '/translate')

if __name__=='__main__':
	app.run(debug=True, host='0.0.0.0')






