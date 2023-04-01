from character import Character
from guesswho import GuessWho
from GuessTree import GuessTree

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
