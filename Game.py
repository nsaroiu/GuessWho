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
from typing import Optional
import pygame
from character import Character
import guesswho
import doctest


def load_screen() -> tuple[pygame.Surface, float, float]:
    """Loads the main menu screen.
    """
    # pygame setup
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    return screen, info.current_w, info.current_h


def rules_screen(screen: pygame.Surface, width: float, height: float) -> None:
    """Loads the rules screen.
    """
    clock = pygame.time.Clock()
    running = True
    dt = 0
    font = pygame.font.SysFont("Arial", 36)
    white = (255, 255, 255)

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return  # return to main menu

        screen.fill("purple")

        txtsurf = font.render("The rules of Guess Who are simple.", True, white)
        screen.blit(txtsurf, (width / 2 - len("The rules of Guess Who are simple.") * 12 / 2, height / 2 - 300))

        txtsurf = font.render("You will be given a character, and you will have to guess who it is.", True, white)
        screen.blit(txtsurf, (
            width / 2 - len("You will be given a character, and you will have to guess who it is.") * 12 / 2,
            height / 2 - 250))

        txtsurf = font.render(
            "You will be given a list of features, and you will have to ask questions to narrow down the list.", True,
            white)
        screen.blit(txtsurf, (width / 2 - len(
            "You will be given a list of features, and you will have to ask questions to narrow down the list.") * 12 / 2,
                              height / 2 - 200))

        txtsurf = font.render("You are expected to type only the given features or a character name (all lowercase)",
                              True, white)
        screen.blit(txtsurf, (width / 2 - len(
            "You are expected to type only the given features or a character name (all lowercase)") * 12 / 2,
                              height / 2 - 150))

        txtsurf = font.render("Click anywhere to continue.", True, white)
        screen.blit(txtsurf, (width / 2 - len("Click anywhere to continue.") * 12 / 2, height / 2 - 100))

        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


def pick_algorithm(screen: pygame.Surface, width: float, height: float) -> str:
    """Picks an algorithm for the AI to use.
    """
    clock = pygame.time.Clock()
    running = True
    dt = 0
    font = pygame.font.SysFont("Arial", 36)
    white = (255, 255, 255)

    button_dict = {}
    algorithm_list = ['CART', 'ID3', 'C4.5', 'Chi-squared', 'variance']

    for i, algorithm in enumerate(algorithm_list):
        # center the button horizontally and offset the y position
        button_dict[algorithm] = TextButton(algorithm,
                                            (width / 2 - len(algorithm) * 12 / 2, -250 + height / 2 + 50 * i))

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in button_dict:
                    if button_dict[button].on_click(event):
                        return button_dict[button].text

        screen.fill("purple")

        for button in button_dict:
            txtsurf = font.render(button_dict[button].text, True, white)
            screen.blit(txtsurf, button_dict[button].rect)

        txtsurf = font.render("Choose your algorithm", True, white)
        screen.blit(txtsurf, (width / 2 - len("Choose your algorithm") * 12 / 2, height / 2 - 300))
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


def pick_character(screen: pygame.Surface, width: float, height: float, dataset: dict[str, Character]) -> \
        tuple[Character, dict[str, Button]]:
    """Picks a character for the player.
    """
    clock = pygame.time.Clock()
    running = True
    dt = 0
    font = pygame.font.SysFont("Arial", 36)
    white = (255, 255, 255)

    button_dict = {}
    char_list = ['alex', 'alfred', 'anita', 'anne', 'bernard', 'bill', 'charles', 'claire',
                 'david', 'eric', 'frans', 'george', 'herman', 'joe', 'maria', 'max', 'paul',
                 'peter', 'philip', 'richard', 'robert', 'sam', 'susan', 'tom']

    # info = pygame.display.Info()  # You have to call this before pygame.display.set_mode()
    screen_width, screen_height = width, height
    add_width = screen_width / 8
    add_height = screen_height / 5
    count = 0

    for character in char_list:
        file = 'images/' + character + '.jpg'
        img = pygame.image.load(file).convert()
        image_size = (130, 140)
        # Scale the image to your needed size
        img = pygame.transform.scale(img, image_size)
        button_dict[character] = Button(img, (20 + add_width * (count % 8),
                                              add_height * (count // 8)), character)
        count += 1

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in button_dict:
                    if button_dict[button].on_click(event):
                        return dataset[button_dict[button].character], button_dict

        screen.fill("purple")

        for button in button_dict:
            screen.blit(button_dict[button].image, button_dict[button].rect)

        load_message(screen, width, height, "Pick any one of the characters by clicking on an image", 0, 100)
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


def load_characters(game: guesswho.GuessWho, screen: pygame.Surface, width: float, height: float,
                    button_dict: dict[str, Button],
                    selected_character: str) -> bool:
    """Main function for the game.
    """
    base_font = pygame.font.SysFont("Arial", 32)
    second_font = pygame.font.SysFont("Arial", 20)
    input_rect = pygame.Rect(width / 2.2 - 50, height * 3 / 5 + 155, 235, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    message1 = ''
    message2 = ''

    active = False

    user_text = ''
    clock = pygame.time.Clock()
    running = True
    dt = 0
    char_list = ['alex', 'alfred', 'anita', 'anne', 'bernard', 'bill', 'charles', 'claire',
                 'david', 'eric', 'frans', 'george', 'herman', 'joe', 'maria', 'max', 'paul',
                 'peter', 'philip', 'richard', 'robert', 'sam', 'susan', 'tom']
    char_stats = {character: True for character in char_list}

    screen_width, screen_height = width, height
    add_width = screen_width / 8
    add_height = screen_height / 5

    # info = pygame.display.Info()  # You have to call this before pygame.display.set_mode()

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        if game.get_winner() is game.player:
            return True
        elif game.get_winner() is game.ai:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos):
                    active = True
                else:
                    active = False
                for button in button_dict:
                    if button_dict[button].on_click(event):
                        char_stats[button_dict[button].character] = not char_stats[button_dict[button].character]
            elif event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        if game.valid_guess(user_text):
                            result = game.player.make_guess(game, user_text)
                            if game.is_character_guess(user_text):
                                if result:
                                    pass
                                else:
                                    message1 = f'You have incorrectly guessed the AI\'s character: {user_text}'
                            else:
                                if result:
                                    message1 = f'The AI\'s character has the {user_text} feature!'
                                else:
                                    message1 = f'The AI\'s character does not have the {user_text} feature'
                            user_text = ''
                            if game.get_winner() is None:
                                guess = game.ai.make_guess(game)
                                if game.is_character_guess(guess):
                                    message2 = f'The AI has guessed that your character is: {guess}'
                                else:
                                    message2 = f'The AI has asked about the {guess} feature'
                        else:
                            message1 = 'Invalid guess. Please try again.'
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode
                else:
                    pass

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("purple")

        for button in button_dict:
            if char_stats[button_dict[button].character]:
                screen.blit(button_dict[button].image, button_dict[button].rect)
            else:
                pass

        # Drawing Rectangle
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(width / 2.2 - 5, height * 3 / 5 - 5, 140, 150))
        screen.blit(button_dict[selected_character].image, (width / 2.2, height * 3 / 5))

        if active:
            color = color_active
        else:
            color = color_inactive

        pygame.draw.rect(screen, color, input_rect)

        txtsurf = base_font.render(user_text, True, (255, 255, 255))
        screen.blit(txtsurf, (width / 2.2 - 49, height * 3 / 5 + 151))

        input_rect.w = max(235, txtsurf.get_width() + 10)

        if message1 is not None:
            load_message(screen, width, height, message1, 400, 150)
        if message2 is not None:
            load_message(screen, width, height, message2, 400, 170)
        # flip() the display to put your work on screen

        features = list(game.features)
        num_features = len(features)
        message_list = ['']
        new_message = ''
        max_char = 55
        i = 0
        for f in features:
            if len(f) + len(message_list[i]) > max_char:
                message_list.append(f + ', ')
                i += 1
            else:
                message_list[i] += f + ', '

        message_list[i] = message_list[i][:-2]

        count = 0
        load_message(screen, width, height, 'Available Features to ask about:', -width / 4 - 50, 100 + count * 20,
                     'red')
        for message in message_list:
            load_message(screen, width, height, message, -width / 4 - 50, 120 + count * 30)
            count += 1
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


def winner_screen(screen: pygame.Surface, width: float, height: float, winner: bool) -> None:
    """Display the winner screen."""
    base_font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    running = True
    dt = 0

    if winner:
        text = 'Congratulations! You have won the game!'
        file = 'images/dog_smile.png'
        img = pygame.image.load(file).convert()
        image_size = (img.get_width() / 2, img.get_height() / 2)
        img = pygame.transform.scale(img, image_size)

    else:
        file = 'images/terminator.png'
        img = pygame.image.load(file).convert()
        text = 'The AI has correctly guessed your character and has won the game. Better luck next time!'
        image_size = (img.get_width() / 2, img.get_height() / 2)
        # img = pygame.transform.scale(img, image_size)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill("purple")
        load_message(screen, width, height, text, 0, 50)
        screen.blit(img, (width / 2 - img.get_width() / 2, height / 2 - 350))

        pygame.display.flip()
        dt = clock.tick(60) / 1000


class Button:
    """A button that can be clicked on.

    Instance Attributes:
        - image: the image of the button
        - rect: the rectangle of the button
        - character: the character that the button represents

    Representation Invariants:
        - self.image is not None
        - self.rect is not None

    """

    def __init__(self, image: pygame.image, position: tuple, character: Optional[str] = None) -> None:
        self.image = image
        self.rect = image.get_rect(topleft=position)
        self.character = character

    def on_click(self, event) -> bool:
        """return True if the button is clicked."""
        if event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class TextButton:
    """A button that can be clicked on.

    Instance Attributes:
        - text: the text of the button
        - rect: the rectangle of the button

    Representation Invariants:
        - self.text is not None
        - self.rect is not None
    """

    def __init__(self, text: str, position: tuple) -> None:
        self.text = text
        self.rect = pygame.Rect(position[0], position[1], 100, 50)

    def on_click(self, event) -> bool:
        """return True if the button is clicked."""
        if event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


def load_message(screen: pygame.surface, width: float, height: float, message: str, add_width: float, add_height: float,
                 colour: str = 'white') -> None:
    """Load the message onto the screen.
    """
    base_font = pygame.font.SysFont("Arial", 26)
    if colour == 'red':
        text_colour = (255, 0, 0)
    else:
        text_colour = (255, 255, 255)
    txtsurf = base_font.render(message, True, text_colour)
    screen.blit(txtsurf, (width / 2 - len(message) * 10 / 2 + add_width, height / 2 + add_height))


def load_character(char_stat: bool, screen: pygame.surface, char_dict: dict[str, pygame.image],
                   character: str, width: float, height: float) -> None:
    """Load the character image onto the screen.
    """
    if char_stat:
        screen.blit(char_dict[character], (width, height))
    else:
        pass


if __name__ == '__main__':
    doctest.testmod()
