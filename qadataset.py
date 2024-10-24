''' Create QADataset and DataLoader'''

import torch
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):
    '''
    A custom dataset class designed for Question Answering (QA) tasks. This class
    inherits from PyTorch's `Dataset` class and is used to handle tokenized data 
    (encodings) and associated labels. It allows the creation of an iterable dataset
    that can be passed to a PyTorch DataLoader for batching and shuffling.
    '''
    def __init__(self, encodings, labels):
        '''
        Initializes the QADataset with the provided tokenized encodings and labels.

        Parameters:
        ----------
        encodings : dict
            The tokenized input data for each example in the dataset.
        labels : list
            The labels associated with each example in the dataset.
        '''
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        '''
        Returns the total number of examples in the dataset.

        Returns:
        -------
        int
            The number of examples in the dataset, which is the length of the labels list.
        '''
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Retrieves an example and its associated label by index.

        Parameters:
        ----------
        idx : int
            The index of the example to retrieve.

        Returns:
        -------
        dict
            A dictionary where the keys correspond to the features (from the encodings),
            and the values are the tokenized data for that example, converted to 
            PyTorch tensors. The 'labels' key holds the label for that example as a tensor.
        '''
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item