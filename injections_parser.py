import csv
import re
from typing import List

replacement_regex: str

def replace(text: str):
    return re.sub(replacement_regex, 'x', text)

with open('sql_list.txt') as f:
    data = f.read()
    sql_actions = list(map(lambda action: action if action == ' ' else action.strip(), data.split('\n')))
    
    escaped_tokens = list(map(lambda token: re.escape(token), sql_actions))

    replacement_regex = '|'.join(escaped_tokens)
    replacement_regex = rf'\b(?!({replacement_regex}))\w+\b'

f.close()
        
with open('SQLiV3.csv', 'r') as f:
    reader = csv.reader(f, delimiter='\n')

    for row in reader:
        replaced = replace(''.join(row))
        print(replaced)

f.close()