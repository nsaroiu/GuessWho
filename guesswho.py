import numpy as np
from character import Character

class GuessWho():

    character_board: np.ndarray
    player1_character: int
    player2_character: int

    def __int__(self, character_board: list[Character]) -> None:
        self.character_board = np.array(character_board)
