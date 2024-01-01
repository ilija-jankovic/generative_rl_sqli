import random
from typing import Dict, List
import string

class WikiSQLParser:
    __COLUMN_STRING_ALPHABET = string.digits + string.ascii_letters + '%-_@$./'

    schema: Dict[str, str]

    def __init__(self, schema: Dict[str, str]):
        self.schema = schema

    def __generate_randomised_example(self, query: str):
        table = random.choice(list(self.schema.keys()))
        columns = self.schema[table]
        
        query = query.replace('[TABLE_NAME]', table)
        
        while '[COLUMN_NAME]' in query:
            column = random.choice(columns)
            query = query.replace('[COLUMN_NAME]', column, 1)

        while '[COMPARISON_VALUE]' in query:
            if random.randint(0, 1) == 0:
                random_str = ''.join(random.choices(self.__COLUMN_STRING_ALPHABET, k=random.randint(1, 20)))
            else:
                random_str = ''.join(random.choices(string.digits, k=random.randint(1, 6)))
            
            query = query.replace('[COMPARISON_VALUE]', random_str, 1)

        return query

    def generate_randomised_examples(self, wikisql_queries: List[str]):
        return [self.__generate_randomised_example(query) for query in wikisql_queries]
