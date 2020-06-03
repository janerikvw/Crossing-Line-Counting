from glob import glob
import os
import json
import re

import basic_entities as entities

lines = {
    'IM05': [
        # Bottom
        ((600, 400), (610, 719)),
        # Right
        ((800, 300), (1200, 250)),
        # Left
        ((0, 250), (350, 380))
    ],
    'IM02': [
        # Bottom left
        ((320, 400), (500, 450)),
        # Bottom right
        ((550, 400), (900, 470)),
        # Middle
        ((400, 200), (850, 260)),
        # Top
        ((450, 50), (900, 100))

    ],
    'IM01': [
        # Bottom
        ((150, 700), (750, 450)),
        # Top
        ((60, 220), (500, 50))
    ],
    'IM03': [
        # Middle all
        ((600, 550), (640, 250)),
        # Single line, top
        ((800, 380), (810, 300)),
        # Small line, right
        ((1000, 390), (980, 600))
    ],
    'IM04': [
        # Top centre
        ((740, 160), (830, 90)),
        # Left centre
        ((750, 350), (920, 480)),
        # Top
        ((200, 200), (520, 0)),
        # Left
        ((20, 250), (300, 720))
    ]
}

def load_full_dataset(path)


if __name__ == '__main__':
    load_all_videos('../data/TUBCrowdFlow')