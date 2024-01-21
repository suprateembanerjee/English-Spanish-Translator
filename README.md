# English to Spanish Translator
A multi-container full stack application containing a Tensorflow (Keras) Transformer model translating English sentences into Spanish, served over a Flask API in the back end and a React + TypeScript + CSS app in the front end.

## Usage

### Docker Compose
By default, this script runs on pre-trained weights downloaded at build time of the backend container. Training the model locally is possible, via steps mentioned towards the end of this document.

`docker compose up --build`

The app is served on `localhost:5173/`

<img width="1354" alt="Demo usage" src="https://github.com/suprateembanerjee/English-Spanish-Translator/assets/26841866/802fce11-4bb0-452f-93cc-0108eb3525aa">

## Specifications

### Model
The model being served is a simple Transformer model (20M parameters) consisting of a single Transformer Encoder and a single Transformer Decoder layer with dropout, trained over 30 epochs.
### Dataset
~120k English and Spanish sentence translations dataset from [ManyThings](www.manythings.org/anki) and [Tatoeba](tatoeba.org).
### Back-end Container(s)
Two Dockerfiles exist: The one we use for `docker compose` and another we use for training the model from scratch. Both contain scripts for installing required dependencies. 

- [The pretrained script](./backend/Dockerfile_pretrained) contains an exclusive script used to download the model weights (in `.keras` format) and source and target vectorization weights (in `.pkl` format)

  `docker build -t translator-pretrained -f dockerfile-pretrained .`

  `docker run -it -p 5000:5000 translator-pretrained python3 src/api.py`

- [The training script](./backend/Dockerfile_train) intiates the training pipeline, which will run at build time.
  
  `docker build -t translator-train -f dockerfile-train .`

  `docker run -it -p 5000:5000 translator-train python3 src/api.py`

The script serves a Flask-based REST API to query the inference model, requiring a JSON object containing the query text in English, source vectorization, target vectorization, and model. The API will return a dictionary with a single key `translated` with a value containing the translated string. The API is served on `LocalHost:5000/translate`.

Example query: `curl -i -H "Content-Type: application/json" -X POST -d '{"text": "I have never heard her speak english", "source_vectorization": "models/english_vectorization.pkl", "target_vectorization": "models/spanish_vectorization.pkl", "model": "models/translator_transformer.keras"}' localhost:5000/translate`

Alternatively, use [demo.py](./backend/demo.py) for querying the backend API. Usage: `python demo.py "I love dancing"`

### Front-end Container
The script serves a React App built using [ViteJS](https://vitejs.dev) containing custom CSS elements for aesthetic purposes. The app makes `fetch` requests based on context and serves the result in a read-only Textbox. The API is served on `LocalHost:5173/`.


## Conclusions

This project explores the different components of a full-stack web application involving deep learning inference. The model itself is not too accurate, as can be expected from a shallow transformer, but it is fairly accurate for the most part and serves the purposes of demonstration. Only 15000 most frequently occurring words were considered for vocabulary, and the Transformer was configured with a sequence length of 20, which is fairly limited. This project, however, showcases the pipeline for training and inference across multiple containers.
