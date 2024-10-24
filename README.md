# Q&A Chatbot Project

This project implements a question-and-answer chatbot using BERT for intent classification. The chatbot takes user questions as input and returns a predicted intent and a corresponding response. The system is structured using FastAPI as the backend, Streamlit for the UI, and a BERT-based model for language understanding.

## Table of Contents
- Overview
- Features
- File Descriptions
- Prerequisites
- Installation
- Usage
- Backend
- Frontend
- API Endpoints
- Model Training
- Contributing
- License

## Overview
The chatbot is designed to:

- Use a pre-trained BERT model for intent classification.
- Convert a JSON-based dataset of intents into a format suitable for model training.
- Serve the model predictions via a FastAPI backend.
- Provide a simple, user-friendly UI using Streamlit.

## Features

- BERT-based Intent Classification: The model uses BERT to classify user inputs into predefined intents.
- JSON to DataFrame Conversion: The dataset is in JSON format and is converted to a pandas DataFrame for easier processing.
- Model Serving with FastAPI: The trained model is served through a REST API using FastAPI.
- Interactive UI with Streamlit: The chatbot interface is built using Streamlit, allowing users to ask questions and receive real-time responses.

## File Descriptions
1. `json_to_df.py`
This script contains a class JsonToDF that converts the intents.json file into a pandas DataFrame. It validates the structure of the JSON file and provides methods for saving the DataFrame as a CSV file.

2. `qadataset.py`
Defines the QADataset class, which prepares the tokenized data and labels for input to the BERT model. It uses the encoded data generated by the tokenizer for model training.

3. `tokenizer.py`
This script contains a function tokenize_data that tokenizes the input patterns from the dataset using BERT's tokenizer and encodes the tags into numerical labels.

4. `main.py`
This script handles model training. It reads the dataset, tokenizes the inputs, and trains a BERT-based sequence classification model. After training, it saves the model and associated metadata (tag-to-label mappings).

5. `predict.py`
Contains the logic for loading the trained model and predicting the intent of user input. It also generates appropriate responses based on the predicted intent.

6. `app.py`
Implements a FastAPI application that serves the chatbot model. It exposes an endpoint (/qa) where users can send their questions and receive a response along with the predicted intent.

7. `app_ui.py`
This is the Streamlit UI for the chatbot. Users can input their questions into a text field, and the app sends the question to the FastAPI backend and displays the predicted response and intent.

## Prerequisites
Before running the project, make sure you have the following installed:

- Python 3.8+
- PyTorch (with CUDA support if available)
- Transformers
- FastAPI
- Uvicorn
- Streamlit
- Pandas

You can install these libraries via pip:
```bash
pip install requirements.txt
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/kayc0des/qabot_model.git
cd qabot_model
```

2. Prepare your dataset in JSON format. Place your intents.json in a json/ folder.

3. Convert the JSON to a DataFrame and save it as a CSV
```bash
python json_to_df.py
```

4. Train the BERT model
```bash
python main.py
```

## Usage

### Backend

1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```
This will launch the API on `http://localhost:8000`

### Frontend

2. Start the Streamlit app for the user interface:
```bash
streamlit run app_ui.py
```
This will open a browser window with the chatbot interface.

## API Endpoints

The FastAPI backend exposes the following endpoint:

- POST /qa: Accepts a JSON payload with a question and returns the predicted intent and response.

    - Request:
    ```json
    {
    "text": "your question here"
    }
    ```

    - Response:
    ```json
    {
    "response": "bot's response",
    "intent": "predicted intent"
    }
    ```

## Model Training

To train the model, ensure your dataset is in CSV format (intents.csv), then run the main.py script to train the model and save it for later use.

### Training Process:
- The model is trained using BERT for sequence classification.
- The tokenizer.py script handles tokenization of input patterns.
- The qadataset.py defines how the data is loaded for training.
- The model is trained using the AdamW optimizer and cross-entropy loss for classification.
- The trained model is saved along with the label mappings for future predictions.

## Contributing
Feel free to fork this repository and contribute by submitting a pull request. We welcome any improvements or additional features! I couldn't add the model since I maxed out on my git lfs😂, so yeah!

## Author
Kingsley Budu
