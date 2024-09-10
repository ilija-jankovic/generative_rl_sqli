import random
from typing import Dict, List
import string

class WikiSQLParser:
    __COLUMN_STRING_ALPHABET = string.digits + string.ascii_letters + '%-_@$./'

    schema: Dict[str, List[str]]

    def __init__(self, schema: Dict[str, str]):
        self.schema = schema

    def __generate_random_string(self):
        return ''.join(random.choices(self.__COLUMN_STRING_ALPHABET, k=random.randint(1, 20)))

    def __generate_randomised_example(self, query: str):
        is_unknown_table = random.randint(0, 1) == 0
        
        table = self.__generate_random_string() if is_unknown_table \
            else random.choice(list(self.schema.keys()))
            
        columns = [
            self.__generate_random_string()
                for _ in range(random.randint(1, 5))
        ] if is_unknown_table else self.schema[table]
        
        query = query.replace('[TABLE_NAME]', table)
        
        while '[COLUMN_NAME]' in query:
            column = random.choice(columns)
                
            query = query.replace('[COLUMN_NAME]', column, 1)

        while '[COMPARISON_VALUE]' in query:
            if random.randint(0, 1) == 0:
                random_str = self.__generate_random_string()
            else:
                random_str = ''.join(random.choices(string.digits, k=random.randint(1, 6)))
            
            query = query.replace('[COMPARISON_VALUE]', random_str, 1)

        return query

    def generate_randomised_examples(self, wikisql_queries: List[str]):
        return [self.__generate_randomised_example(query) for query in wikisql_queries]
