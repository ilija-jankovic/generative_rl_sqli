import random
from typing import Dict, List
import string

__TABLE_COLUMN_ALPHABET = string.digits + string.ascii_letters + '_'

__visible_chars_without_sql_escape = [chr(i) for i in range(32, 127)]
__visible_chars_without_sql_escape.remove('\'')
__visible_chars_without_sql_escape.remove('\\')

schema: Dict[str, List[str]]

def __generate_random_table_column_name():
    return ''.join(random.choices(__TABLE_COLUMN_ALPHABET, k=random.randint(1, 20)))

def __generate_random_string():
    return ''.join(random.choices(__visible_chars_without_sql_escape, k=random.randint(1, 20)))

def __generate_randomised_example(schema: Dict[str, str], query: str):
    is_unknown_table = random.randint(0, 1) == 0
    
    table = __generate_random_table_column_name() if is_unknown_table \
        else random.choice(list(schema.keys()))
        
    columns = [
        __generate_random_table_column_name()
            for _ in range(random.randint(1, 5))
    ] if is_unknown_table else schema[table]
    
    query = query.replace('[TABLE_NAME]', table)
    
    while '[COLUMN_NAME]' in query:
        column = random.choice(columns)
            
        query = query.replace('[COLUMN_NAME]', column, 1)

    while '[COMPARISON_VALUE]' in query:
        if random.randint(0, 1) == 0:
            random_str = f'\'{__generate_random_string()}\''
        else:
            random_str = ''.join(random.choices(string.digits, k=random.randint(1, 6)))
        
        query = query.replace('[COMPARISON_VALUE]', random_str, 1)

    return query

def generate_randomised_examples(
    schema: Dict[str, str],
    queries: List[str],
):
    return [
        __generate_randomised_example(
            schema=schema,
            query=query,
        ) for query in queries
    ]
