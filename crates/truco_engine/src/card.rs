//! Card representations, static rank/envido lookup tables, and fast comparisons.

pub const CARD_RANKS: [u8; 40] = [
    14, 9, 10, 1, 2, 3, 12, 5, 6, 7, // Espadas: 1, 2, 3, 4, 5, 6, 7, 10, 11, 12
    13, 9, 10, 1, 2, 3,  4, 5, 6, 7, // Bastos
     8, 9, 10, 1, 2, 3, 11, 5, 6, 7, // Oros
     8, 9, 10, 1, 2, 3,  4, 5, 6, 7, // Copas
];

pub const CARD_ENVIDO: [u8; 40] = [
    1, 2, 3, 4, 5, 6, 7, 0, 0, 0, // Espadas
    1, 2, 3, 4, 5, 6, 7, 0, 0, 0, // Bastos
    1, 2, 3, 4, 5, 6, 7, 0, 0, 0, // Oros
    1, 2, 3, 4, 5, 6, 7, 0, 0, 0, // Copas
];

pub const CARD_SUITS: [u8; 40] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Espadas (0)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Bastos (1)
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Oros (2)
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // Copas (3)
];

const CARD_NUMBERS: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12];

#[inline(always)]
pub fn compare_cards(c1: u8, c2: u8) -> i8 {
    // ponytail: branchless comparison of static rank table
    let r1 = CARD_RANKS[c1 as usize];
    let r2 = CARD_RANKS[c2 as usize];
    if r1 > r2 {
        1
    } else if r1 < r2 {
        -1
    } else {
        0
    }
}

#[inline(always)]
pub fn card_suit(c: u8) -> u8 {
    CARD_SUITS[c as usize]
}

#[inline(always)]
pub fn card_rank(c: u8) -> u8 {
    CARD_RANKS[c as usize]
}

#[inline(always)]
pub fn card_envido(c: u8) -> u8 {
    CARD_ENVIDO[c as usize]
}

#[inline(always)]
pub fn card_number(c: u8) -> u8 {
    CARD_NUMBERS[(c % 10) as usize]
}
