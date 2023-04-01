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
from GuessTree import GuessTree

from python_ta.contracts import check_contracts


@check_contracts
class GuessWho:
    """ A class representing the state of a game of Guess Who

        Representation Invariants:
            - self.player_board != set()
            - len(self.player_board) == 24

        Instance Attributes:
            - player_board: a set of all characters that the player has not eliminated
            - features: a set of all the valid features that has not been asked about
            - dataset: a dictionary of the characters in the game, where the keys are the character names
            - player: the player who is playing as player 1
            - ai: the player who is playing as player 2
            - _winner: the player who won the game, or None if the game is not yet over
            - _player_turn: boolean representing whether it is player 1's turn or not

    """
    player_board: set[str]
    features: set[str]

    dataset: dict[str, Character]

    player: Player
    ai: AIPlayer

    _winner: Optional[Player]

    _player_turn: bool

    def __init__(self, player_board: set[str], player_character: Character, ai_character: Character, tree: GuessTree,
                 dataset: dict[str, Character], features: set[str], player_turn: bool = True) -> None:

        self.player_board = player_board

        self.dataset = dataset
        self.features = features

        self.player = Player(player_character)
        self.ai = AIPlayer(ai_character, tree)

        self._winner = None

        self._player_turn = player_turn

    def is_player_turn(self) -> bool:
        """Return whether it is the given player's turn.
        """
        return self._player_turn

    def record_move(self, guess: str) -> bool:
        """Record the move made by the corresponding player. Return whether the player's character has the given
        feature or if their character guess was correct.

        Preconditions:
            - not self.is_player_turn() or self.valid_guess(guess)
        """

        # If it's the player's turn, treat it as such
        if self._player_turn:

            # If the given guess is not a valid feature or character name, raise an error
            if not self.valid_guess(guess):
                raise ValueError('invalid feature or character name')

            # If the guess is a character name, check if the guess is the correct character, else remove the guessed
            # character
            if guess in self.dataset:
                if guess != self.ai.get_character().name:
                    self.player_board.remove(guess)
                    return_value = False
                else:
                    self._record_winner(self.player)
                    return_value = True
            else:
                # If the ai's character has the feature, remove all characters that do not have the feature from the
                # game
                if self.ai.get_character().has_feature(guess):
                    for character in self.player_board:
                        if not self.dataset[character].has_feature(guess):
                            self.player_board.remove(character)

                    # Set the return value to true, as the ai's character has the feature
                    return_value = True

                # If the ai's character does not have the feature, remove all characters that have the feature from the
                # game
                else:
                    for character in self.player_board:
                        if self.dataset[character].has_feature(guess):
                            self.player_board.remove(character)

                    # Set the return value to false, as the ai's character does not have the feature
                    return_value = False

        # Else, treat it as the AI's turn
        else:
            # If the guess is a character name, check if the guess is the correct character, and return true if it is
            if guess in self.dataset:
                if guess == self.player.get_character().name:
                    self._record_winner(self.ai)
                    return_value = True
                # In theory, this should never return False, as the AI should never incorrectly guess the player's
                # character
                else:
                    return_value = False

            # Return whether the player's character has the feature
            else:
                return_value = self.player.get_character().has_feature(guess)

        self._player_turn = not self._player_turn
        return return_value

    def valid_guess(self, guess: str) -> bool:
        """Return whether the given guess is a valid guess.
        """
        return guess in self.features or guess in self.dataset

    def _record_winner(self, winner: Player) -> None:
        """Record the winner of the game.
        """
        self._winner = winner

    def get_winner(self) -> Optional[Player]:
        """Return the winner of the game.
        """
        return self._winner

class Player:
    """ An abstract class representing a player in a game of Guess Who."""

    _character: Character

    def __init__(self, character: Character) -> None:
        """Initialize this player."""
        self._character = character

    def get_character(self) -> Character:
        """Return the character that this player has chosen."""
        return self._character

    def make_guess(self, game: GuessWho) -> None:
        """Make a move in the given game of Guess Who.
        Preconditions:
            - game.is_player_turn()
        """

        # Print available features and characters, then ask the player to make a guess
        print(f'Available features to ask about: {game.features}\n-----------------')
        print(f'Remaining valid characters: {game.player_board}\n-----------------')

        # Ask the player to make a valid guess
        guess = ''
        while not game.valid_guess(guess):
            guess = input("Make a character guess or ask about a feature (case-sensitive): ")

            if not game.valid_guess(guess):
                print('Invalid feature or character name. Please enter a valid feature or character name.')

            print('-----------------')

        # Record the guess in the game
        has_feature = game.record_move(guess)

        # Print the result of the guess
        if guess in game.features:
            print(f'The character has the {guess} feature!' if has_feature else f'The character does not have the {guess} feature.')
        else:
            print('You correctly guessed the character!' if has_feature else 'That is not the correct character.')

        print('=================')


class AIPlayer(Player):
    """ A class representing an AI player in a game of Guess Who."""

    tree: GuessTree

    def __init__(self, character: Character, tree: GuessTree) -> None:
        """Initialize this AI player."""
        super().__init__(character)
        self.tree = tree

    def make_guess(self, game: GuessWho) -> None:
        """Make a move in the given game of Guess Who.
        Preconditions:
            - not game.is_player_turn()
        """
        # Get the guess from the tree
        guess = self.tree.get_guess()

        # Record the guess
        has_feature = game.record_move(guess)

        # Update the tree given the whether the player's character has the feature
        self.tree.traverse_tree(has_feature)
