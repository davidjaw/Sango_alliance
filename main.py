from typing import List, Tuple
import numpy as np
import cv2
import tkinter as tk
import argparse
import os
import yaml
from utils import ScreenCap
from tk_control import TkController
import time
from cnocr import CnOcr
from utils import NeuralNetworks


def parse_args():
    # Read configurations from the config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_screen', type=int, default=0)
    parser.add_argument('--num_screens', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--toolbar_left', action='store_true')
    # Parse known command-line arguments
    args, remaining = parser.parse_known_args()

    # Override config values with command-line argument values
    for arg in remaining:
        key, value = arg.split('=')
        config[key] = value
    return config


def main(args: dict):
    # find the program's location by matching the icon
    sc = ScreenCap(
        display_screen=args['display_screen'],
        num_screens=args['num_screens'],
        icon_path=args['icon_path'],
        toolbar_left=args['toolbar_left'],
        debug=args['debug']
    )
    # check if icon is in screen
    loc = sc.match_icon()
    while loc is None:
        time.sleep(.333)
        loc = sc.match_icon()
    sc.adjust_region(loc)
    # load models and configs
    ocr = CnOcr(rec_model_name='chinese_cht_PP-OCRv3')
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    nn = NeuralNetworks(
        num_class=10,
        weights_path='digit.pth',
        config=config['nn_config']
    )
    # create tk controller
    root = tk.Tk()
    tk_controller = TkController(root, sc, ocr, args['debug'], model=nn, out_file=config['out_file'])
    tk_controller.mainloop()


if __name__ == '__main__':
    main(parse_args())
