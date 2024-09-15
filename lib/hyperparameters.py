STATE_SIZE = 1024
ACTION_SIZE = 64
EMBEDDING_DIM = 128

# T << episode length pg. 5 (PPO paper).
T = 10

EPOCHS = 10
BATCH_SIZE = 1024
MINIBATCH_SIZE = 512

# This value (epsilon) is based on best performing clipping strategy
# in Table 1, pg. 7 (PPO paper).
PPO_PROBABILITY_RATIO_CLIP_THRESHOLD = 0.2
PPO_SUCCESSFUL_POLICY_PROBABILITY = 0.5
PPO_DEMONSRATION_SAMPLING_PROBABILITY = 0.1
PPO_SUCCESSFUL_BUFFER_SIZE = 4096

GAMMA = 0.9995
INITIAL_ACTOR_LEARNING_RATE = 0.001
INITIAL_CRITIC_LEARNING_RATE = 0.002
L2_WEIGHT = 0.0001

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.9
ADAM_EPSILON=1e-5

LR_SCHEDULE_DECAY_STEPS=200
LR_SCHEDULE_DECAY_RATE=0.9

ACTOR_LSTM_UNITS = 128
ACTOR_DENSE_UNITS = 128

# NOTE: State needs a few tokens of leeway for sectioning.
assert(STATE_SIZE >= 8)

# Assert some chance for actor interaction with environments.
assert(PPO_SUCCESSFUL_POLICY_PROBABILITY <= 0.8)

assert(BATCH_SIZE >= MINIBATCH_SIZE)
assert(BATCH_SIZE % MINIBATCH_SIZE == 0)

# TODO: Figure out what exactly n actors means. Is it batch size?
#
# M <= NT from Algorithm 1 in pg. 5, where M is minibatch size.
# N = 1 as there are no parallel actors.
#assert(MINIBATCH_SIZE <= T)