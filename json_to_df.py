''' This Script Contains a class that converts a JSON File into a pandas dataframe '''


import json
import os


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
                self.json = json.load(file)
            except json.JSONDecodeError:
                raise TypeError(f"The file '{file_path}' does not contain valid JSON.")