from guesswho import GuessWho, Player, AIPlayer
import data
from character import Character
import random
from GuessTree import GuessTree
import Game


def run_game() -> None:
    """Run a game of Guess Who.
    """

    with open('./data/guess_who.csv') as file:
        features = file.readline().split(',')[:-1]
        dataset = {}  # Dictionary mapping character names to character objects
        character_list = []  # List of character objects
        tree = GuessTree.read_guess_tree('./data/CART_tree.pkl')

        for line in file:
            line = line.split(',')

            # Get a list of the character's corresponding features as ints
            character_features_int = line[:-1]

            # Get the character name
            character_name = line.pop()[:-1]

            # Convert the list of ints to a list of booleans
            character_features_bool = [True if feature == '1' else False for feature in character_features_int]

            # Create a dictionary of the character's features mapping the feature name to the boolean value
            character_features = {}
            for i in range(len(features)):
                character_features[features[i]] = character_features_bool[i]

            # Create character object
            character = Character(character_name, character_features)

            # Add the character to the dataset
            dataset[character_name] = character
            character_list.append(character)

    character_names = [character.name for character in dataset.values()]
    chosen_player_character = input(
        f"Choose a character from the following list (case-sensitive): {character_names}\n")
    while chosen_player_character not in character_names:
        print("Invalid character name. Please try again.")
        print('-----------------')
        chosen_player_character = input(
            f"Choose a character from the following list (case-sensitive): {character_names}\n")

    print('\n==================================\n')

    # Initialize the players
    player = Player(dataset[chosen_player_character])
    ai = AIPlayer(random.choice(character_list), tree)

    # Initialize the game
    game = GuessWho(player, ai, dataset, set(features))

    while game.get_winner() is None:
        if game.is_player_turn():
            print('It is your turn.\n-----------------')
            game.player.make_guess(game)

        else:
            print('It is the AI\'s turn.\n-----------------')
            guess = game.ai.make_guess(game)
            if guess in game.dataset:
                print(f'The AI has guessed that your character is: {guess}.\n\n==================================\n')
            else:
                print(f'The AI has asked about the {guess} feature.\n\n==================================\n')

    if game.get_winner() is game.player:
        print('Congratulations! You have won the game!')
    else:
        print(
            f'The AI has correctly guessed your character and has won the game. Better luck next time!\n\nThe AI\'s character was: {game.ai.get_character().name}')


def run_visualization():
    """Run a visualization of a game of Guess Who.
    """
    with open('./data/guess_who.csv') as file:
        features = file.readline().split(',')[:-1]
        dataset = {}  # Dictionary mapping character names to character objects
        character_list = []  # List of character objects

        screen, width, height = Game.load_screen()
        Game.rules_screen(screen, width, height)

        algorithm = Game.pick_algorithm(screen, width, height)
        tree_file = './data/' + algorithm + '_tree.pkl'
        tree = GuessTree.read_guess_tree(tree_file)

        for line in file:
            line = line.split(',')

            # Get a list of the character's corresponding features as ints
            character_features_int = line[:-1]

            # Get the character name
            character_name = line.pop()[:-1]

            # Convert the list of ints to a list of booleans
            character_features_bool = [True if feature == '1' else False for feature in character_features_int]

            # Create a dictionary of the character's features mapping the feature name to the boolean value
            character_features = {}
            for i in range(len(features)):
                character_features[features[i]] = character_features_bool[i]

            # Create character object
            character = Character(character_name, character_features)

            # Add the character to the dataset
            dataset[character_name] = character
            character_list.append(character)

    chosen_player_character, button_dict = Game.pick_character(screen, width, height, dataset)
    player = Player(chosen_player_character)
    ai = AIPlayer(random.choice(character_list), tree)

    # Initialize the game
    game = GuessWho(player, ai, dataset, set(features))
    winner = Game.load_characters(game, screen, width, height, button_dict, chosen_player_character.name)
    Game.winner_screen(screen, width, height, winner)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_visualization()
