from .environment import Environment


class InitialTransitionsFactory:
    num_transitions: int
    env: Environment

    def __init__(self, num_transitions: int, env: Environment):
        