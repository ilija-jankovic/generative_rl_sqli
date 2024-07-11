import tensorflow as tf

# T << episode length pg. 5.
T = 10
PARALLEL_ACTORS = 1
timestep = 0
gamma = 0.999
value_model = None
policy_model = None
probability_ratio_clip_threshold = 0.0001

def value_estimator():
    pass

def stochastic_policy():
    pass

def calculate_advantage(
    initial_timestep,
    terminal_timestep,
    first_state,
    last_state,
    rewards
):
    global T, gamma, value_model

    assert(len(rewards) == T)
    assert(terminal_timestep > initial_timestep)

    advantage = -value_model(first_state)

    timestep_window = terminal_timestep - initial_timestep
    for t in range(initial_timestep, terminal_timestep):
        advantage += pow(
                gamma,
                timestep_window + t - terminal_timestep
            ) * rewards[t]

    advantage += pow(gamma, timestep_window) * value_model(last_state)

    return tf.convert_to_tensor(advantage)

def calculate_clipped_probability_ratios(probability_ratios, advantages):
    assert(len(probability_ratios) == T)
    assert(len(advantages) == T)

    clipped = tf.clip_by_value(
        probability_ratios,
        1.0 - probability_ratio_clip_threshold,
        1.0 + probability_ratio_clip_threshold
    )

    return tf.minimum(probability_ratios * advantages, clipped * advantages)

def clipped_surrogate_loss(
    y,
    y_old,
    first_state,
    last_state,
    rewards
):
    global T

    assert(len(y_old) == T)
    assert(len(y) == T)

    probability_ratios = y / y_old
    advantages = [
        calculate_advantage(timestep + t, timestep + T, first_state, last_state, rewards)
            for t in range(timestep, timestep + T)
    ]
    advantages = tf.convert_to_tensor(advantages)

    minimums = calculate_clipped_probability_ratios(probability_ratios, advantages)

    return tf.reduce_mean(minimums)
