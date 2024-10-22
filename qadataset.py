''' Create QADataset and DataLoader'''

import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW

tokenize_data = __import__('tokenizer').tokenize_data

data_path = 'data/intents.csv'
tokenizer = BertTokenizer.from_pretrained('bet-base-uncased')
df = pd.read_csv(data_path)


class QADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

if __name__ == '__main__':
    encoded_data, labels, num_labels = tokenize_data(df, tokenizer=tokenizer)
    dataset = QADataset(encoded_data, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    epochs = 30
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
            loss.backward()  # Backward pass

            optimizer.step()  # Update parameters

        print(f"Epoch {epoch+1} completed with loss {loss.item()}")
    
    model.eval()
    
    