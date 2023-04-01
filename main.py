from guesswho import GuessWho
import data
from character import Character
import random
from GuessTree import GuessTree

def run_game() -> None:
    """Run a game of Guess Who.
    """

    with open('./data/guess_who.csv') as file:
        features = file.readline().split(',')
        dataset = {} # Dictionary mapping character names to character objects
        character_list = [] # List of character objects
        tree = GuessTree.read_guess_tree('./data/CART_tree.pkl')

        for line in file:
            line = line.split(',')

            # Get a list of the character's corresponding features as ints
            character_features_int = line[:-1]

            # Get the character name
            character_name = line.pop()

            # Convert the list of ints to a list of booleans
            character_features_bool = [bool(feature) for feature in character_features_int]

            # Create a dictionary of the character's features mapping the feature name to the boolean value
            character_features = {}
            for i in range(len(features)):
                character_features[features[i]] = character_features_bool[i]

            # Create character object
            character = Character(character_name, character_features)

            # Add the character to the dataset
            dataset[character_name] = character
            character_list.append(character)

    game = GuessWho(random.choice(character_list), random.choice(character_list), tree, dataset, set(features))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_game()
