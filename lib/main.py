import numpy as np
import requests
from rl.ddpg import DDPG
from rl.pre_training_environment import PreTrainingEnvironment
from rl.environment import Environment
from rl.server_environment import ServerEnvironment

ACTION_SIZE = 20
STATE_SIZE = 10

visible_chars = [chr(i) for i in range(32, 127)]
numbers = [str(i) for i in range(0, 10)]

with open('sql_list.txt', 'r') as f:
    data = f.read()
f.close()

sql_list = data.split('\n')

with open('tables.txt', 'r') as f:
    data = f.read()
f.close()

tables = list(map(lambda table: table + ' ', data.split('\n')))

with open('columns.txt', 'r') as f:
    data = f.read()
f.close()

columns = list(map(lambda column: column + ' ', data.split('\n')))

# sql_list must account for the leading segment of the action space.
#
# This is due to the pre-processing of existing SQL injections for
# pretraining.
dictionary = sql_list + numbers + tables + columns

injected_payloads = []

terminated = False
state: np.ndarray

environment: Environment

def __toggle_environment(is_pre_training: bool):
    '''
    Must be called before running DQN.

    `state` must be defined before calling this function.
    '''
    global environment

    environment = PreTrainingEnvironment(dictionary, action_size=ACTION_SIZE, state_size=STATE_SIZE) \
        if is_pre_training \
        else ServerEnvironment(dictionary, action_size=ACTION_SIZE, state_size=STATE_SIZE,
                               send_request_callback=lambda payload: requests.get(f'http://127.0.0.1:5000/pages?prodLine={payload}'))
        #res = requests.post('http://localhost.proxyman.io:3000/rest/user/login', data={
        #    'email': payload
        #})

__toggle_environment(is_pre_training=True)

ddpg = DDPG(environment)
ddpg.run()