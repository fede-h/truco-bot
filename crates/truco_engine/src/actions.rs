//! Action definitions and bitmask helpers matching the Python Action enum.

pub const ACTION_PLAY_CARD_0: u8 = 0;
pub const ACTION_PLAY_CARD_1: u8 = 1;
pub const ACTION_PLAY_CARD_2: u8 = 2;
pub const ACTION_PLAY_CARD_DOWN_0: u8 = 3;
pub const ACTION_PLAY_CARD_DOWN_1: u8 = 4;
pub const ACTION_PLAY_CARD_DOWN_2: u8 = 5;

pub const ACTION_ENVIDO: u8 = 10;
pub const ACTION_REAL_ENVIDO: u8 = 11;
pub const ACTION_FALTA_ENVIDO: u8 = 12;
pub const ACTION_QUIERO_ENVIDO: u8 = 13;
pub const ACTION_NO_QUIERO_ENVIDO: u8 = 14;

pub const ACTION_TRUCO: u8 = 20;
pub const ACTION_RETRUCO: u8 = 21;
pub const ACTION_VALE_CUATRO: u8 = 22;
pub const ACTION_QUIERO_TRUCO: u8 = 23;
pub const ACTION_NO_QUIERO_TRUCO: u8 = 24;
pub const ACTION_IR_AL_MAZO: u8 = 25;

pub const NUM_ACTIONS: usize = 26;

#[inline(always)]
pub fn is_card_action(a: u8) -> bool {
    a <= 5
}

#[inline(always)]
pub fn is_envido_action(a: u8) -> bool {
    (10..=14).contains(&a)
}

#[inline(always)]
pub fn is_truco_action(a: u8) -> bool {
    (20..=25).contains(&a)
}
