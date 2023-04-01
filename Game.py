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
from collections.abc import Callable
from typing import Any, Optional
import pandas as pd
import numpy as np
import pygame
from character import Character
import player
from GuessTree import GuessTree
import guesswho

from python_ta.contracts import check_contracts


def load_screen() -> tuple[pygame.Surface, float, float]:
    """Loads the main menu screen.
    """
    # pygame setup
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    return screen, info.current_w, info.current_h


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
                print(event.pos)
                for button in button_dict:
                    if button_dict[button].on_click(event):
                        return dataset[button_dict[button].character], button_dict

        screen.fill("purple")

        for button in button_dict:
            screen.blit(button_dict[button].image, button_dict[button].rect)

        txtsurf = font.render("Pick any one of the characters by clicking on an image", True, white)
        screen.blit(txtsurf, (width / 4, height * 3 / 4))
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


def load_characters(game_state: guesswho.GuessWho, screen: pygame.Surface, width: float, height: float,
                    button_dict: dict[str, Button],
                    character: str) -> None:
    """Main function for the game.
    """
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
    count = 0
    button_dict = {}
    for character in char_list:
        file = 'images/' + character + '.jpg'
        img = pygame.image.load(file).convert()
        image_size = (130, 140)
        # Scale the image to your needed size
        img = pygame.transform.scale(img, image_size)
        button_dict[character] = Button(img, (20 + add_width * (count % 8),
                                              add_height * (count // 8)), character)
        count += 1

    # info = pygame.display.Info()  # You have to call this before pygame.display.set_mode()

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in button_dict:
                    if button_dict[button].on_click(event):
                        char_stats[button_dict[button].character] = not char_stats[button_dict[button].character]

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("purple")

        for button in button_dict:
            if char_stats[button_dict[button].character]:
                screen.blit(button_dict[button].image, button_dict[button].rect)
            else:
                pass
        screen.blit(button_dict[character].image, (width / 4, height * 3 / 4))
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


class Button:
    """A button that can be clicked on."""

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


def load_character(char_stat: bool, screen: pygame.surface, char_dict: dict[str, pygame.image],
                   character: str, width: float, height: float) -> None:
    """Load the character image onto the screen.
    """
    if char_stat:
        screen.blit(char_dict[character], (width, height))
    else:
        pass


if __name__ == '__main__':
    pass
    screen, width, height = load_screen()
    load_characters('', screen, width, height, {}, 'tom')
