from guesswho import GuessWho, Player, AIPlayer
from character import Character
import random
from GuessTree import GuessTree
import Game

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import statistics


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


def run_visualization() -> None:
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


def tree_data() -> None:
    algorithms = ['ID3', 'CART', 'C4.5', 'Chi-squared', 'variance']

    trees = [GuessTree.read_guess_tree('./data/' + algorithm + '_tree.pkl') for algorithm in algorithms]

    # Height of each tree
    tree_heights = {algorithms[i]: trees[i].get_height() for i in range(len(algorithms))}

    # List of heights of each leaf in each tree
    tree_leaf_heights = {algorithms[i]: trees[i].get_heights() for i in range(len(algorithms))}

    # Average height of each leaf in each tree
    tree_average_heights = {algorithm: sum(tree_leaf_heights[algorithm]) / 24 for algorithm in algorithms}

    # Average height of each leaf in each tree if using optimized guessing
    tree_average_optimized_heights = {algorithms[i]: trees[i].sum_optimized_heights() / 24 for i in
                                      range(len(algorithms))}

    # Median height of the leaves in each tree
    tree_median_leaf_heights = {algorithm: statistics.median(tree_leaf_heights[algorithm]) for algorithm in algorithms}

    # Mode height of the leaves in each tree
    tree_mode_leaf_heights = {algorithm: statistics.mode(tree_leaf_heights[algorithm]) for algorithm in algorithms}

    # Make three graphs that show the best and worst case leaf heights, the different average leaf heights, and compare
    # mean leaf heights to optimized mean leaf heights
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Best Case Leaf Height vs Worst Case Leaf Height', 'Different Averages of Leaf Height', 'Mean Leaf Height vs Mean Optimized Leaf Height'), horizontal_spacing=0.075)

    # Best vs Worst Case Leaf Height graph
    fig.add_trace(go.Bar(name='Best Case Leaf Height', x=algorithms, y=[min(tree_leaf_heights[algorithm]) for algorithm in algorithms]), row=1, col=1)
    fig.add_trace(go.Bar(name='Worst Case Leaf Height', x=algorithms, y=[tree_heights[algorithm] for algorithm in algorithms]), row=1, col=1)

    # Different Averages of Leaf Height graph
    fig.add_trace(go.Bar(name='Mean Leaf Height', x=algorithms, y=[tree_average_heights[algorithm] for algorithm in algorithms]), row=1, col=2)
    fig.add_trace(go.Bar(name='Median Leaf Height', x=algorithms, y=[tree_median_leaf_heights[algorithm] for algorithm in algorithms]), row=1, col=2)
    fig.add_trace(go.Bar(name='Mode Leaf Height', x=algorithms, y=[tree_mode_leaf_heights[algorithm] for algorithm in algorithms]), row=1, col=2)

    # Mean Leaf Height vs Mean Optimized Leaf Height graph
    fig.add_trace(go.Bar(name='Mean Leaf Height', x=algorithms, y=[tree_average_heights[algorithm] for algorithm in algorithms]), row=1, col=3)
    fig.add_trace(go.Bar(name='Mean Optimized Leaf Height', x=algorithms, y=[tree_average_optimized_heights[algorithm] for algorithm in algorithms]), row=1, col=3)

    # Set the titles of the axes and define the range of the y-axis for each graph
    for i in range(1, 4):
        fig['layout'][f'xaxis{i}']['title'] = 'Algorithm Name'
        fig['layout'][f'yaxis{i}']['title'] = 'Leaf Height'
        fig['layout'][f'yaxis{i}']['range'] = [0, 17]

    # Set the margins of the graphs
    fig['layout']['margin'] = dict(l=0, r=0, t=50, b=50)

    fig.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #run_visualization()
    tree_data()
