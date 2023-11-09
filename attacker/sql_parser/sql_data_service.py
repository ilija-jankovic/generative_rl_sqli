import csv
import os
from typing import List

from .sql_injection_parser import encode_payloads

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

    def load_encoded_injections(self):
        sql_csv_path = os.path.join(self.__dirname, '../../SQLiV3.csv')
        sqlmap_payloads_path = os.path.join(self.__dirname,'../sqlmap-log/127.0.0.1/attempted-payloads.txt')

        allowed_path = os.path.join(self.__dirname, '../../sql_list.txt')
        reserved_path = os.path.join(self.__dirname, '../../reserved_sql.txt')
        
        encoded = encode_payloads(sql_csv_path, is_csv=True, sql_list_path=allowed_path, reserved_sql_list_path=reserved_path) + \
            encode_payloads(sqlmap_payloads_path, is_csv=False, sql_list_path=allowed_path, reserved_sql_list_path=reserved_path)
        
        return [[int(index_string) for index_string in token.split(',')] for token in encoded]

    def load_columns(self):
        return self.__read_lines('../../columns.txt')
    
    def load_tables(self):
        return self.__read_lines('../../tables.txt')