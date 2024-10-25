from enum import Enum


class PretrainActorType(Enum):
    PRETRAIN = 1
    '''
    Runs behavioural cloning for actor pretraining.
    '''
    
    LOAD_PRETRAINED = 2
    '''
    Loads saved pretrained actor weights.
    '''

    NO_PRETRAIN = 3
    '''
    Does not pretrain, i.e., demonstrations are not
    used.
    '''
