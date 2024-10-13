from lib.pretrain_actor_type import PretrainActorType


STATE_SIZE = 1024
ACTION_SIZE = 80
EMBEDDING_DIM = 128

PRETRAIN_ACTOR_TYPE = PretrainActorType.NO_PRETRAIN
PRETRAINING_STEPS = 3000
PRETRAINING_LEARNING_RATE = 0.0001

INITIAL_EPISODE_LENGTH = 5
MAX_EPISODE_EXTENSION = 10

# T << episode length pg. 5 (PPO paper).
T = 10

EPOCHS = 3
BATCH_SIZE = 512
MINIBATCH_SIZE = 256

PPO_SUCCESSFUL_BATCH_SIZE = 256
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

GAMMA = 0.999
INITIAL_ACTOR_LEARNING_RATE = 0.0003
INITIAL_CRITIC_LEARNING_RATE = 0.0006
L2_WEIGHT = 0.0001

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.9
ADAM_EPSILON=1e-5

LR_SCHEDULE_DECAY_STEPS=10000
LR_SCHEDULE_DECAY_RATE=0.9

ACTOR_LSTM_UNITS = 64
ACTOR_DENSE_UNITS = 128

# Cannot easily be a global due to lambda layer loading
# with Keras.
#
# Implied value of 2.0.
#
# Do not uncomment unless integrating with PPO actor
# lambda layer.
#
# Make sure to update in PPO reporter as well if used.
#
# ACTOR_SOFTMAX_TEMPERATURE = 2.0

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