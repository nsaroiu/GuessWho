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


def load_characters(game_state: guesswho.GuessWho) -> None:
    """Main function for the game.
    """
    import pygame

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    dt = 0

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    button_dict = {}
    char_list = ['alex', 'alfred', 'anita', 'anne', 'bernard', 'bill', 'charles', 'claire',
                 'david', 'eric', 'frans', 'george', 'herman', 'joe', 'maria', 'max', 'paul',
                 'peter', 'philip', 'richard', 'robert', 'sam', 'susan', 'tom']
    char_stats = {character: True for character in char_list}

    info = pygame.display.Info()  # You have to call this before pygame.display.set_mode()
    screen_width, screen_height = info.current_w, info.current_h
    add_width = screen_width / 8
    add_height = screen_height / 4.5
    count = 0

    for character in char_list:
        file = 'images/' + character + '.jpg'
        img = pygame.image.load(file).convert()
        image_size = (150, 150)
        # Scale the image to your needed size
        img = pygame.transform.scale(img, image_size)
        button_dict[character] = Button(img, (10 + add_width * (count % 8),
                                              10 + add_height * (count // 8)), character)
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
                        print('clicked', button_dict[button].character, 'button')
                        char_stats[button_dict[button].character] = not char_stats[button_dict[button].character]

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("purple")

        # pygame.draw.circle(screen, "red", player_pos, 40)
        #
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_w]:
        #     player_pos.y -= 300 * dt
        # if keys[pygame.K_s]:
        #     player_pos.y += 300 * dt
        # if keys[pygame.K_a]:
        #     player_pos.x -= 300 * dt
        # if keys[pygame.K_d]:
        #     player_pos.x += 300 * dt
        # info = pygame.display.Info()  # You have to call this before pygame.display.set_mode()
        # screen_width, screen_height = info.current_w, info.current_h
        # add_width = screen_width / 8
        # add_height = screen_height / 4.5
        # count = 0
        # for character in ['alex', 'alfred', 'anita', 'anne', 'bernard', 'bill', 'charles', 'claire',
        #                   'david', 'eric', 'frans', 'george', 'herman', 'joe', 'maria', 'max', 'paul',
        #                   'peter', 'philip', 'richard', 'robert', 'sam', 'susan', 'tom']:
        #     # if game_state.character_board[character]:
        #     load_character(False, screen, char_dict, character, 10 + add_width * (count % 8),
        #                    10 + add_height * (count // 8))
        #     count += 1
        for button in button_dict:
            if char_stats[button_dict[button].character]:
                screen.blit(button_dict[button].image, button_dict[button].rect)
            else:
                pass

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
    load_characters([])
