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

    def __parse_encoded_tokens(self, lines: List[str]):
        encoded_injections: List[List[int]] = []

        for payload in lines:
            encoded_injections.append([])

            for token in payload.split(','):
                encoded_injections[-1].append(int(token))

        return encoded_injections

    def load_encoded_injections(self):
        data = self.__read_lines('../../parsed_injections_indexed.txt')
        return self.__parse_encoded_tokens(data)

    def load_sql_list(self):
        return self.__read_lines('../../sql_list.txt')

    def load_columns(self):
        return self.__read_lines('../../columns.txt')
    
    def load_tables(self):
        return self.__read_lines('../../tables.txt')
