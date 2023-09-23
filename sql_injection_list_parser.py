import csv
import re
from typing import List

not_allowed_reserved_syntax: List[str]
replacement_regex: str
parsed_injections: str = ''

def is_injection_allowed(injection: str):
    for token in not_allowed_reserved_syntax:
        if token.lower() in injection.lower():
            return False
    return True

def replace(text: str):
    return re.sub(replacement_regex, '[VARIABLE]', text, flags=re.IGNORECASE)

print('Parsing SQLiV3.csv...')

with open('sql_list.txt') as f:
    data = f.read()
    sql_actions = list(map(lambda action: action if action == ' ' else action.strip(), data.split('\n')))
    
    escaped_tokens = list(map(lambda token: re.escape(token), sql_actions))

    replacement_regex = '|'.join(escaped_tokens)
    replacement_regex = rf'\b(?!({replacement_regex}))\w+\b'
f.close()

with open('reserved_sql.txt', 'r') as f:
    data = f.read()
    reserved_syntax = data.split('\n')
    not_allowed_reserved_syntax = list(set(reserved_syntax) - set(sql_actions))
f.close()

with open('SQLiV3.csv', 'r') as f:
    reader = csv.reader(f, delimiter='\n')

    for row in reader:
        injection = ''.join(row)
        if not is_injection_allowed(injection):
            continue

        replaced = replace(injection)
        parsed_injections += f'{replaced}\n'
f.close()

with open('parsed_injections.txt', 'w') as f:
    # Remove last newline character with rstrip().
    f.write(parsed_injections.rstrip())
f.close()

print('Parsed injections written to parsed_injections.txt')