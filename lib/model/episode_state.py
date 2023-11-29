class EpisodeState:
    initial_frames: int
    batch_size: int

    __frames: int
    __frames_since_last_episode: int = 0

    def __init__(self, initial_frames: int, batch_size: int):
        self.initial_frames = initial_frames
        self.batch_size = batch_size
        
        self.next_episode()

    def next_episode(self):
        self.__frames = self.initial_frames
        self.__frames_since_last_episode = 0

    def next_frame(self):
        self.__frames_since_last_episode += self.batch_size

    def has_episode_ended(self):
        return self.__frames_since_last_episode >= self.__frames
    
    def extend_episode(self):
        self.__frames += self.initial_frames + self.__frames_since_last_episode - self.__frames 
