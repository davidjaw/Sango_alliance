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
from utils import NeuralNetworks
from utils import left_align_and_pad, CustomOCRNet
import torch


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
    config = yaml.load(open('ocr_config.yaml', 'r'), Loader=yaml.FullLoader)
    data = yaml.load(open('data.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ocr = CustomOCRNet(class_num=len(data), batch_size=1, device=device)
    st_dict = torch.load(f'weights/ocr_epoch_300.pth')
    ocr.load_state_dict(st_dict)
    ocr.to(device)
    ocr.eval()
    transform_func = left_align_and_pad(config['nn_config']['img_size'])
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    nn = NeuralNetworks(
        pt_path='nn.pt',
        config=config['nn_config']
    )
    # create tk controller
    root = tk.Tk()
    tk_controller = TkController(root, sc, ocr, args['debug'], model=nn, out_file=config['out_file'], transform=transform_func)
    tk_controller.mainloop()


if __name__ == '__main__':
    main(parse_args())
