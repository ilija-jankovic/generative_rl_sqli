import csv
import re
from typing import List

VARIABLE_TOKEN = '[VARIABLE]'

sql_actions: List[str]
escaped_sql_actions: List[str]

not_allowed_reserved_syntax: List[str]
parsed_injections: str = ''

def is_injection_allowed(injection: str):
    for token in not_allowed_reserved_syntax:
        if token.lower() in injection.lower():
            return False
    return True

def replace_with_constraints(text: str, irreplacable_tokens: List[str]):
    tokens_regex = '|'.join(irreplacable_tokens)
    return re.sub(rf'\b(?!({tokens_regex}))\w+\b', VARIABLE_TOKEN, text, flags=re.IGNORECASE)

def replace(text: str, to_replace: str, replacement: str):
    return re.sub(to_replace, replacement, text, flags=re.IGNORECASE)

def encode_with_action_indicies(injection: str, comma_placeholder: str):
    """
    Ensure `sql_actions` does not contain `comma_placeholder`.
    """
    for i in range(len(sql_actions)):
        escaped_action = escaped_sql_actions[i]
        injection = replace(injection, escaped_action, f'{i}{comma_placeholder}')
    
    encoded_actions = replace(injection, re.escape(VARIABLE_TOKEN), f'-1{comma_placeholder}')
    
    # Remove last comma.
    return encoded_actions[:-1]

def encode_all_with_action_indicies(injections: List[str]):
    COMMA_PLACEHOLDER = 'a'

    for i in range(len(injections)):
        injections[i] = encode_with_action_indicies(injections[i], COMMA_PLACEHOLDER)
    
    return [injection.replace(COMMA_PLACEHOLDER, ',') for injection in injections]

print('Parsing SQLiV3.csv...')

with open('sql_list.txt') as f:
    data = f.read()

    sql_actions = list(map(
        lambda action: action \
        if action == ' ' \
            else action.strip(), data.split('\n')))
    
    escaped_sql_actions = list(map(
        lambda token: re.escape(token), sql_actions))
f.close()

with open('reserved_sql.txt', 'r') as f:
    data = f.read()
    reserved_syntax = data.split('\n')

    # Parse intersection between all SQL keywords and keywords
    # in the action space.
    not_allowed_reserved_syntax = \
        list(set(reserved_syntax) - set(sql_actions))
f.close()

with open('SQLiV3.csv', 'r') as f:
    reader = csv.reader(f, delimiter='\n')

    for row in reader:
        injection = ''.join(row)
        if not is_injection_allowed(injection):
            continue

        replaced = replace_with_constraints(injection, escaped_sql_actions)
        parsed_injections += f'{replaced}\n'
f.close()

with open('parsed_injections.txt', 'w') as f:
    # Remove last newline character with rstrip().
    f.write(parsed_injections.rstrip())
f.close()

print('Parsed injections written to parsed_injections.txt')
print('Encoding injections into action indicies...')

with open('parsed_injections_indexed.txt', 'w') as f:
    indexed_injections = encode_all_with_action_indicies(parsed_injections.split('\n'))

    # Remove any incorrectly formatted encodings incase of incorrect processing
    # such as imperfect regex.
    for i in range(len(indexed_injections) - 1, -1, -1):
        for token in indexed_injections[i].split(','):
            if not str.isnumeric(token) and token != '-1':
                del indexed_injections[i]
                break

    indexed_injections = [injection + '\n' for injection in indexed_injections]

    # Remove last newline character with rstrip().
    indexed_injections[-1] = indexed_injections[-1].rstrip()
    
    f.writelines(indexed_injections)
f.close()

print('Encoded parsed injections into parsed_injections_indicies.txt')