''' Train Moddel '''

import pandas as pd
from transformers import BertTokenizer


def tokenize_data(data, tokenizer):
    '''
    Tokenizes the patterns (input) and encodes the tags as int
    
    Args:
        data: pd dataframe
        tokenizer: an instance of the tokenizer
    '''
    encoded_data = tokenizer(list(data['Patterns']), padding=True, truncation=True, return_tensors='pt')
    tag_to_label = {tag: idx for idx, tag in enumerate(data['Tag'].unique())}
    num_labels = len(tag_to_label)
    labels = data['Tag'].map(tag_to_label).values
    
    return encoded_data, labels, num_labels