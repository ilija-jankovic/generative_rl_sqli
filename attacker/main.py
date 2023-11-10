#!/usr/local/bin/python
import numpy as np
import requests
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from rl.ddpg import DDPG
from rl.environment import Environment

ACTION_SIZE = 100
STATE_SIZE = 10000

IP = '127.0.0.1'

visible_chars = [chr(i) for i in range(32, 127)]

data_service = SQLDataService()

columns = data_service.load_columns()
tables = data_service.load_tables()

sql_tokens = data_service.load_sql_tokens()
token_blacklist = data_service.load_sql_blacklist()

payloads = data_service.load_payload_files(IP)

dictionary = sql_tokens + tables + columns + visible_chars
dictionary.sort(reverse=True)

print('Encoding payloads...')
encoded_injections = TokenParser(dictionary, token_blacklist, payloads).parse()
print('Payloads encoded.')

environment = Environment(
    dictionary, action_size=ACTION_SIZE, state_size=STATE_SIZE,
    encoded_injections=encoded_injections,
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
    for injection in encoded_injections:
        decoded = [dictionary[i] for i in injection]
        print(''.join(decoded))

print_decoded_injections()

def main():
    ddpg = DDPG(environment)
    ddpg.run()

if __name__ == '__main__':
    main()