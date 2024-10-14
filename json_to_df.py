''' This Script Contains a class that converts a JSON File into a pandas dataframe '''


class JSON_to_DF():
    ''' Converts JSON to DF '''

    def __init__(self, path):
        '''
        Constructor method
        
        Args:
            path: Path to JSON File
         
        Returns:
            Class Instance
        '''

        # Check file extension