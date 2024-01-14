class DDPGHyperparameters:

    gamma: float

    # Parameter for updating targets.
    tau: float

    actor_learning_rate: float
    critic_learning_rate: float
    embedding_size: int
    buffer_size: int
    batch_size: int
    alpha_scalar: float
    starting_adaptive_sigma: float
    starting_adaptive_delta: float
    psi: float
    action_size: int
    state_size: int

    def __init__(
            self,
            gamma: float,
            tau: float,
            actor_learning_rate: float,
            critic_learning_rate: float,
            embedding_size: int,
            buffer_size: int,
            batch_size: int,
            alpha_scalar: float,
            starting_adaptive_sigma: float,
            starting_adaptive_delta: float,
            psi: float,
            action_size: int,
            state_size: int,
        ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.embedding_size = embedding_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha_scalar = alpha_scalar
        self.starting_adaptive_sigma = starting_adaptive_sigma
        self.starting_adaptive_delta = starting_adaptive_delta
        self.psi = psi
        self.action_size = action_size
        self.state_size = state_size