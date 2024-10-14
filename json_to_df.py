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
                self.validate_data(self.data)
            except json.JSONDecodeError:
                raise TypeError(f"The file '{file_path}' does not contain valid JSON.")
    
    def validate_data(self, data):
        '''
        Validates self.data to have a key intents and intents is a list of dictionaries
        and every dict has three keys tag, patterns and responses
        
        Args:
            data: self.data
        
        Returns:
            Calls convert to_df
        '''
    
    def to_df(self, data):
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