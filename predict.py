import torch
import pickle
from transformers import BertTokenizer

# Global variables
model = None
tokenizer = None
tag_to_label = None
label_to_tag = None
device = None

def load_model(model_path='model/qa_model.pkl'):
    global model, tokenizer, tag_to_label, label_to_tag, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model and data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    tag_to_label = model_data['tag_to_label']
    label_to_tag = {v: k for k, v in tag_to_label.items()}
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model.to(device)
    model.eval()

def predict_intent(question):
    """
    Predicts an intent (question) and maps the prediction to a tag

    Args:
        question (str): User input string

    Returns:
        str: Predicted tag
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
    
    return predicted_tag

if __name__ == '__main__':
    # Load the model
    load_model()
    
    # Main loop for user interaction
    print("Intent Prediction System")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'quit':
            break
        
        predicted_tag = predict_intent(question)
        print(f"Predicted Intent: {predicted_tag}")

    print("Thank you for using the Intent Prediction System!")