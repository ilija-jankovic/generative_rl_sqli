STATE_SIZE = 1024
ACTION_SIZE = 64
EMBEDDING_DIM = 128

# T << episode length pg. 5 (PPO paper).
T = 10

EPOCHS = 10
BATCH_SIZE = 512
MINIBATCH_SIZE = 256

PPO_SUCCESSFUL_BATCH_SIZE = 32
'''
The number of successful trajectories to replay on every learning
step.

Strictly less than batch size.
'''

ENVIRONMENT_BATCH_SIZE = BATCH_SIZE - PPO_SUCCESSFUL_BATCH_SIZE
'''
The number of on-policy transitions to collect from environment
on every learning step.

`Environment batch size + successful batch size = batch size`
'''

# This value (epsilon) is based on best performing clipping strategy
# in Table 1, pg. 7 (PPO paper).
PPO_PROBABILITY_RATIO_CLIP_THRESHOLD = 0.2
PPO_SUCCESSFUL_BUFFER_SIZE = 256

GAMMA = 0.9995
INITIAL_ACTOR_LEARNING_RATE = 0.0001
INITIAL_CRITIC_LEARNING_RATE = 0.0002
L2_WEIGHT = 0.0001

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON=1e-5

LR_SCHEDULE_DECAY_STEPS=100
LR_SCHEDULE_DECAY_RATE=0.9

ACTOR_LSTM_UNITS = 128
ACTOR_DENSE_UNITS = 128

# NOTE: State needs a few tokens of leeway for sectioning.
assert(STATE_SIZE >= 8)

assert(BATCH_SIZE > PPO_SUCCESSFUL_BATCH_SIZE)
assert(BATCH_SIZE % PPO_SUCCESSFUL_BATCH_SIZE == 0)

assert(BATCH_SIZE >= MINIBATCH_SIZE)
assert(BATCH_SIZE % MINIBATCH_SIZE == 0)

# TODO: Figure out what exactly n actors means. Is it batch size?
#
# M <= NT from Algorithm 1 in pg. 5, where M is minibatch size.
# N = 1 as there are no parallel actors.
#assert(MINIBATCH_SIZE <= T)