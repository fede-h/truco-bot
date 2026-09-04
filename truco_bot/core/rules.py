from truco_bot.core.card import Card


def calculate_envido(cards: list[Card]) -> int:
    by_suit: dict[int, list[int]] = {}
    for card in cards:
        by_suit.setdefault(card.suit.value, []).append(card.envido_value)
    
    max_envido = 0
    for values in by_suit.values():
        if len(values) >= 2:
            values.sort(reverse=True)
            score = values[0] + values[1] + 20
        else:
            score = values[0]
        max_envido = max(max_envido, score)
        
    return max_envido

def compare_cards(c1: Card, c2: Card) -> int:
    # return 1, -1, or 0 based on rank directly.
    if c1.truco_rank > c2.truco_rank:
        return 1
    if c1.truco_rank < c2.truco_rank:
        return -1
    return 0

def resolve_hand(trick_results: list[int], mano: int = 0) -> int | None:
    # trick_results: 1 (P0 win), -1 (P1 win), 0 (parda)
    p0 = trick_results.count(1)
    p1 = trick_results.count(-1)
    
    # 2 trick wins immediately
    if p0 >= 2: return 0
    if p1 >= 2: return 1
    
    if len(trick_results) >= 2:
        # First trick decided, second parda -> first winner wins
        if trick_results[0] != 0 and trick_results[1] == 0:
            return 0 if trick_results[0] == 1 else 1
        # First trick parda, second decided -> second winner wins
        if trick_results[0] == 0 and trick_results[1] != 0:
            return 0 if trick_results[1] == 1 else 1

    if len(trick_results) == 3:
        # Third trick parda -> first trick winner wins
        if trick_results[2] == 0 and trick_results[0] != 0:
            return 0 if trick_results[0] == 1 else 1
        # T1 parda, T2 parda, T3 decided -> T3 winner wins
        if trick_results[0] == 0 and trick_results[1] == 0 and trick_results[2] != 0:
            return 0 if trick_results[2] == 1 else 1
        # All parda -> mano wins
        if trick_results.count(0) == 3:
            return mano
            
    return None

def calculate_falta_envido_points(
            score_p0: int, 
            score_p1: int, 
            max_score: int = 30
        ) -> int:
    return max_score - max(score_p0, score_p1)
