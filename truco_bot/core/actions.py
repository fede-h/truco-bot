# ponytail: minimal action enum, simple helper functions
from enum import IntEnum


class Action(IntEnum):
    PLAY_CARD_0 = 0
    PLAY_CARD_1 = 1
    PLAY_CARD_2 = 2
    PLAY_CARD_DOWN_0 = 3
    PLAY_CARD_DOWN_1 = 4
    PLAY_CARD_DOWN_2 = 5
    
    ENVIDO = 10
    REAL_ENVIDO = 11
    FALTA_ENVIDO = 12
    QUIERO_ENVIDO = 13
    NO_QUIERO_ENVIDO = 14
    
    TRUCO = 20
    RETRUCO = 21
    VALE_CUATRO = 22
    QUIERO_TRUCO = 23
    NO_QUIERO_TRUCO = 24
    IR_AL_MAZO = 25

def is_card_action(a: Action) -> bool:
    return 0 <= a.value <= 5

def is_envido_action(a: Action) -> bool:
    return 10 <= a.value <= 14

def is_truco_action(a: Action) -> bool:
    return 20 <= a.value <= 25
