import torch
import pickle
import pandas as pd
from transformers import BertTokenizer
import random

# Global variables
model = None
tokenizer = None
tag_to_label = None
label_to_tag = None
device = None
df = None

def load_model(model_path='model/qa_model.pkl', data_path='data/intents.csv'):
    global model, tokenizer, tag_to_label, label_to_tag, device, df
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model and data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    tag_to_label = model_data['tag_to_label']
    label_to_tag = {v: k for k, v in tag_to_label.items()}
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    model.to(device)
    model.eval()

def get_response(tag):
    global df
    responses = df[df['Tag'] == tag]['Responses'].values[0]
    return random.choice(responses.split('|')) if '|' in responses else responses

def predict_and_respond(question):
    """
    Predicts an intent (question), maps the prediction to a tag, and provides a response

    Args:
        question (str): User input string

    Returns:
        tuple: Predicted tag and corresponding response
    """
    global model, tokenizer, label_to_tag, device
    
    # Ensure model is loaded
    if model is None:
        load_model()
    
    # Tokenize and predict
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_tag = label_to_tag[predictions.item()]
    
    # Get response for the predicted tag
    response = get_response(predicted_tag)
    
    return predicted_tag, response

if __name__ == '__main__':
    # Load the model
    load_model()
    
    # Main loop for user interaction
    print("Q&A System")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nYou: ")
        if question.lower() == 'quit':
            break
        
        predicted_tag, response = predict_and_respond(question)
        print(f"Bot: {response}")
        print(f"(Predicted Intent: {predicted_tag})")

    print("Thank you for using the Q&A System!")