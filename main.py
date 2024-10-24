import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
import pickle

tokenize_data = __import__('tokenizer').tokenize_data
QADataset = __import__('qadataset').QADataset

data_path = 'data/intents.csv'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = pd.read_csv(data_path)

if __name__ == '__main__':
    encoded_data, labels, num_labels, tag_to_label = tokenize_data(df, tokenizer=tokenizer)
    dataset = QADataset(encoded_data, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    epochs = 50
    for epoch in range(epochs):
        for batch in dataloader:
            # Move batch data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch+1} completed with loss {loss.item()}")
    
    model.eval()
    
    # Save the entire model, not just the state dict
    model_data = {
        'model': model,
        'num_labels': num_labels,
        'tag_to_label': tag_to_label
    }
    with open('model/qa_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved successfully.")