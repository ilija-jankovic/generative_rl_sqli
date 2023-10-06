import numpy as np
import requests
from rl.sql_data_service import SQLDataService
from rl.ddpg import DDPG
from rl.environment import Environment

ACTION_SIZE = 20
STATE_SIZE = 20

visible_chars = [chr(i) for i in range(32, 127)]
numbers = [str(i) for i in range(0, 10)]

data_service = SQLDataService()

sql_list = data_service.load_sql_list()
columns = data_service.load_columns()
tables = data_service.load_tables()

# sql_list must account for the leading segment of the action space.
#
# This is due to the pre-processing of existing SQL injections for
# pretraining.
dictionary = sql_list + numbers + columns + tables

encoded_injections = data_service.load_encoded_injections(dictionary)

environment = Environment(
    dictionary, action_size=ACTION_SIZE, state_size=STATE_SIZE,
    encoded_injections=encoded_injections, sql_syntax=sql_list,
    columns=columns, tables=tables,
    send_request_callback=lambda payload: requests.get(f'http://127.0.0.1:5000/pages?prodLine={payload}'))
        #res = requests.post('http://localhost.proxyman.io:3000/rest/user/login', data={
        #    'email': payload
        #})
                                         
state: np.ndarray

def print_decoded_injections():
    '''
    Use to verify that the predefined encoded injection list is correctly
    encoded.
    '''
    for injection in encoded_injections:
        decoded = ['[VARIABLE]' if i == -1 else dictionary[i] for i in injection]
        print(''.join(decoded))

ddpg = DDPG(environment)
ddpg.run()
