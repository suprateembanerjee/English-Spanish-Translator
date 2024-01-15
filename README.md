# English to Spanish Translator
A Transformer-based English to Spanish translator application, containerized and served for inference over a REST API

## Usage

### Utilizing the pre-trained model
**Step 1**: Build the docker container serving the inference API <br/>
- Installs requirements<br/>
- Downloads pre-trained models<br/>

`docker build -t translator-pretrained -f dockerfile-pretrained .`

**Step 2**: Start the inference API server <br/>
- Maps container port 5000 to host port 5000 used to make POST requests for inference

`docker run -it -p 5000:5000 translator-pretrained python3 src/api.py`

**Step 3**: Query the API
- Refer to demo.py for an example of the API call

`python demo.py "I have never heard her speak english"`

- Alternatively, for a more raw API query,

`curl -i -H "Content-Type: application/json" -X POST -d '{"text": "I have never heard her speak english", "source_vectorization": "models/english_vectorization.pkl", "target_vectorization": "models/spanish_vectorization.pkl", "model": "models/translator_transformer.keras"}' localhost:5000/translate`

### Training a new model from scratch

**Step 1**: Build the docker container, training taking place at build time <br/>
- Installs requirements
- Trains a Transformer model as per spec mentioned in dockerfile (feel free to tune)
- Saves weights

`docker build -t translator-train -f dockerfile-train .`

**Step 2**: Start the inference API server <br/>
- Maps container port 5000 to host port 5000 used to make POST requests for inference

`docker run -it -p 5000:5000 translator-train python3 src/api.py`

**Step 3**: Query the API
- Refer to demo.py for an example of the API call

`python demo.py "I have never heard her speak english"`

- Alternatively, for a more raw API query,

`curl -i -H "Content-Type: application/json" -X POST -d '{"text": "I have never heard her speak english", "source_vectorization": "models/english_vectorization.pkl", "target_vectorization": "models/spanish_vectorization.pkl", "model": "models/translator_transformer.keras"}' localhost:5000/translate`
