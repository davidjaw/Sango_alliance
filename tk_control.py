from typing import List, Tuple
import os
import numpy as np
import cv2
import tkinter as tk
from utils import ScreenCap
import keyboard
from prepro_test import preprocess_name_patch


class TkController(object):
    def __init__(
            self,
            tk_root: tk.Tk,
            screen_cap: ScreenCap,
            **kwargs,
    ):
        self.tk_root = tk_root
        self.screen_cap = screen_cap
        self.kwargs = kwargs
        # create widgets
        self.info_label = None
        self.detect_button = None
        self.create_widgets()

    def create_widgets(self):
        # create button
        self.detect_button = tk.Button(self.tk_root, text="Detect", command=self.detect_button_pressed, height=2)
        self.detect_button.pack(fill='x', padx=10, expand=True)
        # create label
        self.info_label = tk.Label(self.tk_root, text="")
        self.info_label.pack()
        # bind keyboard listener
        keyboard.add_hotkey('space', self.detect_button_pressed)

    def detect_button_pressed(self):
        img = self.screen_cap.grab_screen()
        self.screen_cap.save_image(img)
        # preprocess_name_patch(img)

    def mainloop(self):
        self.tk_root.geometry("300x100")
        self.tk_root.mainloop()
