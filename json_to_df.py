''' This Script Contains a class that converts a JSON File into a pandas dataframe '''


import json
import os
import pandas as pd


class JsonToDF():
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
        Converts the JSON data into a pandas DataFrame.

        Returns:
            pd.DataFrame: The JSON data as a DataFrame 
            with 'Tag', 'Patterns', and 'Responses' columns.
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
    
    def save_df(self, filename, directory):
        '''
        Saves the DataFrame to a CSV file in the specified directory.
        
        Args:
            filename (str): Name of the output CSV file (without extension)
            directory (str): Directory where the CSV should be saved
        
        Raises:
            ValueError: If DataFrame is not created before saving
        '''
        if not hasattr(self, 'df'):
            raise ValueError("No DataFrame to save. Convert JSON to DataFrame first.")
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.df.to_csv(os.path.join(directory, f"{filename}.csv"), index=False)
        
if __name__ == '__main__':
    path = 'json/intents.json'
    jsontodf = JsonToDF(path)
    df = jsontodf.to_df()
    jsontodf.save_df(filename='intents', directory='data')