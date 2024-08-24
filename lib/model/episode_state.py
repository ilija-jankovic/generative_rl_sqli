class EpisodeState:
    initial_frames: int

    __frames: int
    __frames_since_last_episode: int
    
    __episode: int

    @property
    def frames_since_last_episode(self):
        return self.__frames_since_last_episode

    @property
    def episode(self):
        return self.__episode

    def __init__(self, initial_frames: int):
        self.initial_frames = initial_frames
        self.__frames_since_last_episode = 0
        
        self.__episode = 0
        
        self.next_episode()

    def next_episode(self):
        self.__frames = self.initial_frames
        self.__frames_since_last_episode = 0
        
        self.__episode += 1

    def next_frame(self):
        self.__frames_since_last_episode += 1

    def has_episode_ended(self):
        return self.__frames_since_last_episode >= self.__frames
    
    def extend_episode(self):
        self.__frames += self.initial_frames
        