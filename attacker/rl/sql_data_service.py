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
    
    def __ensure_escaped(self, data: List[List[int]], dictionary: List[str], prefix: str):
        '''
        Assumes dictionary contains a quote, double quote, space, hash, and double dash.
        '''

        # Note: ensure below condition is changed if prefix_indicies length
        # is altered.
        prefix_indicies = [dictionary.index(prefix), dictionary.index(' ')]
    
        # TODO: Ensure double dash if not a MySQL server (known from grey box information).
        suffix_index = dictionary.index('#')

        for i in range(len(data)):
            line = data[i]
            if len(line) < 2:
                continue

            if line[0] != prefix_indicies[0] or line[1] != prefix_indicies[1]:
                data[i] = prefix_indicies + data[i]
            
            if line[-1] != suffix_index:
                data[i].append(suffix_index)
    
    def __escape_sql(self, data: List[str], dictionary: List[str]):
        '''
        Ensures two copies of each line start with a quote or
        double quote, and that they end with a comment based
        on the SQL server.
        '''
        single_quote_data = data
        double_quote_data = data.copy()

        self.__ensure_escaped(single_quote_data, dictionary, '\'')
        self.__ensure_escaped(double_quote_data, dictionary, '"')

        return single_quote_data + double_quote_data        

    def __parse_encoded_tokens(self, lines: List[str]):
        encoded_injections: List[List[int]] = []

        for payload in lines:
            encoded_injections.append([])

            for token in payload.split(','):
                encoded_injections[-1].append(int(token))

        return encoded_injections

    def load_encoded_injections(self, dictionary: List[str]):
        data = self.__read_lines('../../parsed_injections_indexed.txt')
        #data = self.__parse_encoded_tokens(data)
        #return self.__escape_sql(data, dictionary)
        return self.__parse_encoded_tokens(data)

    def load_sql_list(self):
        return self.__read_lines('../../sql_list.txt')

    def load_columns(self):
        return self.__read_lines('../../columns.txt')
    
    def load_tables(self):
        return self.__read_lines('../../tables.txt')
