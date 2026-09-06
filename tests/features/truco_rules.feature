Feature: Argentine Truco Core Rules
  As a player
  I want the core rules to be correctly enforced
  So that the game plays correctly

  Scenario: Parda in first trick is decided by second trick
    Given a new hand where Mano wins the second trick after a parda in the first trick
    When the hand is resolved
    Then Mano wins the hand

  Scenario: Envido scoring with figure cards
    Given a player is dealt a hand with 10 of Copas, 11 of Copas, and 2 of Bastos
    When the player calculates their envido
    Then the envido score should be 20

  Scenario: Refusing Truco ends hand and awards points
    Given a game is dealt
    When player 0 calls Truco
    And player 1 refuses Truco
    Then player 0 wins the hand with 1 points
