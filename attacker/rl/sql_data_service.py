import csv
import os
from typing import List

class SQLDataService: 
    __dirname: str

    def __init__(self):
        self.__dirname = os.path.dirname(__file__)
    
    def __read_lines(self, relative_path: str):
        path = os.path.join(self.__dirname, relative_path)

        with open(path, 'r') as f:
            data = f.read()
        f.close()

        return data.split('\n')
    
    def __encode_row(self, row: List[str], dictionary: List[str]):
        combined_row = ''.join(row)
        return [dictionary.index(char) if char in dictionary else -1 for char in combined_row]

    def load_encoded_injections(self, dictionary: List[str]):
        payloads: List[str] = []

        rows = self.__read_lines('../../SQLiV3.csv')
        reader = csv.reader(rows, delimiter='\n')

        for row in reader:
            payloads.append(self.__encode_row(row, dictionary))

        rows = self.__read_lines('../sqlmap-log/127.0.0.1/attempted-payloads.txt')
        for row in rows:
            payloads.append(self.__encode_row(row, dictionary))

        return payloads

    def load_columns(self):
        return self.__read_lines('../../columns.txt')
    
    def load_tables(self):
        return self.__read_lines('../../tables.txt')
