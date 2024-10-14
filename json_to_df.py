''' This Script Contains a class that converts a JSON File into a pandas dataframe '''


import json
import os
import pandas as pd


class JSON_to_DF():
    ''' Converts JSON to DF '''

    def __init__(self, file_path):
        '''
        Constructor method
        
        Args:
            path: Path to JSON File
         
        Returns:
            Class Instance
        '''

        # Check file extension
        if not file_path.endswith('.json'):
            raise TypeError(f"The file '{file_path}' is not a JSON file.")

        with open(file_path, 'r') as file:
            try:
                self.data = json.load(file)

                if not len(self.data.keys()) == 1:
                    raise ValueError(f"More than one key found")

                data_key = list(self.data.keys())[0]

                if not data_key == 'intents':
                    raise ValueError(f"intents key not found")
                
                # Check every intent to make sure they each have three keys
                # ['tag', 'patterns', 'responses']
                intent_keys = ['tag', 'patterns', 'responses']
                
                for intent in self.data['intents']:
                    if not len(intent.keys()) == 3:
                        raise ValueError(f"Expected three keys for each intent")
                    if not list(intent.keys()) == intent_keys:
                        raise ValueError(f"Each Intent in Intents shouls have three keys (tag, patterns and responses)")

                print('All checks passed')

            except json.JSONDecodeError:
                raise TypeError(f"The file '{file_path}' does not contain valid JSON.")
    
    def to_df(self):
        '''
        Converts data to a pandas dataframe
        
        Args:
            Data: self.data
        
        Return:
            Df : Pandas Dataframe
        '''
        # Convert the intents into a pandas DataFrame
        data = []

        for intent in self.data['intents']:
            tag = intent['tag']
            patterns = ", ".join(intent['patterns'])
            responses = ", ".join(intent['responses'])
            data.append([tag, patterns, responses])

        # Create the DataFrame
        self.df = pd.DataFrame(data, columns=['Tag', 'Patterns', 'Responses'])

        return self.df
    
    def save_df(self, dir):
        '''
        Saves df into the specified directory
        
        Args:
            dir: Directory to save dataframe
        '''
        
if __name__ == '__main__':
    path = 'json/intents.json'
    jsontodf = JSON_to_DF(path)
    df = jsontodf.to_df()
    print(df.head())