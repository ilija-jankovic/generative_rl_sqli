class DDPGHyperparameters:

    gamma: float

    # Parameter for updating targets.
    tau: float

    actor_learning_rate: float
    critic_learning_rate: float
    embedding_size: int
    buffer_size: int
    batch_size: int
    starting_stddev: float
    alpha_scalar: float
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
    psi: float
    temperature: float
    priority_weight: float
    rollout_weight: float
    l2_weight: float
    action_size: int
    state_size: int
    prefix: str
    suffix: str
    constant_stddev: bool

    def __init__(
            self,
            gamma: float,
            tau: float,
            actor_learning_rate: float,
            critic_learning_rate: float,
            embedding_size: int,
            batch_size: int,
            starting_stddev: float,
            psi: float,
            temperature: float,
            n_step_rollout: int,
            priority_weight: float,
            rollout_weight: float,
            l2_weight: float,
            action_size: int,
            state_size: int,
            prefix: str,
            suffix: str,
            constant_stddev: bool,
            alpha_scalar: float = None,
            epsilon_start: float = None,
            epsilon_decay: float = None,
            epsilon_min: float = None
        ) -> None:
        '''
        If `constant_stddev` is true, the standard deviation of noise perturbation
        is always `starting_stddev`.
        
        Otherwise, the standard deviation begins at `starting_stddev`, and decays
        logarithmically based on the epsilon parameters.

        `alpha_scalar`, `epsilon_start`, `epsilon_decay`, and `epsilon_min` must
        all be set if `constant_stddev` is `False`.
        '''
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.starting_stddev = starting_stddev
        self.psi = psi
        self.temperature = temperature
        self.n_step_rollout = n_step_rollout
        self.priority_weight = priority_weight
        self.rollout_weight =rollout_weight
        self.l2_weight = l2_weight
        self.action_size = action_size
        self.state_size = state_size
        self.prefix = prefix
        self.suffix = suffix
        self.constant_stddev = constant_stddev
        self.alpha_scalar = alpha_scalar
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min