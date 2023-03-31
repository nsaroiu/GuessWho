"""Copyright and Usage Information
===============================
Copyright (c) 2023 Prashanth Shyamala and Nicholas Saroiu
All Rights Reserved.
This program is the property of Prashanth Shyamala and Nicholas Saroiu.
All forms of distribution of this code, whether as given or with any changes,
are expressly prohibited.
For more information on copyright for CSC111 materials, please consult our Course Syllabus.
"""
from __future__ import annotations
from typing import Any, Optional
import pickle
import pandas as pd
import numpy as np
from character import Character
import player
import GuessTree as gt

from python_ta.contracts import check_contracts


@check_contracts
class GuessWho:
    """ A class representing the state of a game of Guess Who

        Representation Invariants:
            - self.character_board != {}
            - len(self.character_board) == 24

        Instance Attributes:
            - character_board: a dictionary of the characters in the game, where the keys are the character names
            and the values are booleans of whether the character is still in the game or not
            - player1_character: the character that player 1 has chosen
            - player2_character: the character that player 2 has chosen
            - player_turn: the player whose turn it is

    """
    character_board: dict[str, bool]
    player1_character: Character
    player2_character: Character
    player_turn: player.Player

    def __init__(self, character_board: dict[str, bool], player1: str, player2: str, player1_features: dict[str, bool],
                 player2_features: dict[str, bool]) -> None:
        self.character_board = character_board
        self.player1_character = Character(player1, player1_features)
        self.player2_character = Character(player2, player2_features)

    def is_player_turn(self, player: player.Player) -> bool:
        """Return whether it is the given player's turn.
        """
        return self.player_turn == player


class Player:
    """ An abstract class representing a player in a game of Guess Who."""

    def make_guess(self, game: GuessWho) -> str:
        """Make a move in the given game of Guess Who.
        Preconditions:
            - game.is_player_turn(self)
        """
        raise NotImplementedError


class AIPlayer(Player):
    """ A class representing an AI player in a game of Guess Who."""
    tree: gt.GuessTree

    def make_guess(self, game: GuessWho) -> str:
        """Make a move in the given game of Guess Who.
        Preconditions:
            - game.is_player_turn(self)
        """
        raise NotImplementedError
