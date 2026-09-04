"""Card representations, ranks (1-14), suits, and comparisons."""
from dataclasses import dataclass
from enum import IntEnum


# simple enum for suits
class Suit(IntEnum):
    ESPADAS = 1
    BASTOS = 2
    OROS = 3
    COPAS = 4

@dataclass(frozen=True, slots=True)
class Card:
    number: int
    suit: Suit
    truco_rank: int
    envido_value: int

    def __str__(self) -> str:
        return f"{self.number} de {self.suit.name.capitalize()}"
        
    def __repr__(self) -> str:
        return self.__str__()

# precompute everything to avoid runtime logic overhead
def _get_truco_rank(number: int, suit: Suit) -> int:
    if number == 1 and suit == Suit.ESPADAS: return 14
    if number == 1 and suit == Suit.BASTOS: return 13
    if number == 7 and suit == Suit.ESPADAS: return 12
    if number == 7 and suit == Suit.OROS: return 11
    if number == 3: return 10
    if number == 2: return 9
    if number == 1 and suit in (Suit.COPAS, Suit.OROS): return 8
    if number == 12: return 7
    if number == 11: return 6
    if number == 10: return 5
    if number == 7 and suit in (Suit.COPAS, Suit.BASTOS): return 4
    if number == 6: return 3
    if number == 5: return 2
    if number == 4: return 1
    raise ValueError(f"Invalid card: {number} de {suit.name}")

def _get_envido_value(number: int) -> int:
    return 0 if number >= 10 else number

def create_card(number: int, suit: Suit) -> Card:
    return Card(
        number=number,
        suit=suit,
        truco_rank=_get_truco_rank(number, suit),
        envido_value=_get_envido_value(number)
    )

ALL_CARDS = tuple(
    create_card(n, s) 
    for s in Suit 
    for n in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
)
