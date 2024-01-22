import random
from typing import Dict, List


class ContextualTemplatePopulator:
    
    schema: Dict[str, List[str]]

    def __init__(self, schema: Dict[str, str]) -> None:
        self.schema = schema

    def __generate_randomised_example(self, payload: str):
        table = random.choice(list(self.schema.keys()))

        columns = self.schema[table]
        columns = columns + ['NULL']

        payload = payload.replace('[TABLE_NAME]', table)
        
        while '[COLUMN_NAME]' in payload:
            column = random.choice(columns)
            payload = payload.replace('[COLUMN_NAME]', column, 1)

        while '[COLUMN_NAME_NOT_NULL]' in payload:
            column = random.choice(columns[:-1])
            payload = payload.replace('[COLUMN_NAME_NOT_NULL]', column, 1)

        return payload

    def generate_randomised_examples(self, templates: List[str], count: int = 10000):
        templates = [random.choice(templates) for _ in range(count)]

        return [self.__generate_randomised_example(payload) for payload in templates]
        
