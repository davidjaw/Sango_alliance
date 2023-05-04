import mss
import numpy as np
import cv2
from typing import List, Tuple, Dict
import os
import torch
from torchvision import transforms


class NeuralNetworks(object):
    def __init__(
            self,
            pt_path: str,
            config: Dict,
            device: torch.device | None = None,
            name_dict_path: str | None = None
    ):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.jit.load(pt_path)
        self.model = model.to(self.device)
        self.config = config
        self.img_size = config['img_size']
        self.name_dict_path = name_dict_path
        self.class_to_idx = None
        self.idx_to_class = None

    def predict(self, image) -> np.ndarray:
        data_preprocess = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # convert to float tensor
        image = image.astype(np.float32)
        if np.max(image) > 1:
            image /= 255.
        image = torch.from_numpy(image)
        image = image.permute(0, 3, 1, 2)
        image = data_preprocess(image)
        image = image.to(self.device)
        with torch.no_grad():
            pred = self.model(image)
            _, pred = torch.max(pred, dim=1)
            return pred.cpu().numpy()


class ScreenCap(object):
    def __init__(
            self,
            display_screen: int = 0,
            num_screens: int = 2,
            save_dir: str = 'tmp/sc',
            debug: bool = False,
            icon_path: str = 'dictionary/icon.png',
            toolbar_left: bool = False,
            **kwargs,
    ):
        # use mss to get the base coordinate of the i-th screen
        self.capture_coord: dict | None = None
        self.save_dir = save_dir
        self.display_screen = display_screen
        self.num_screens = num_screens
        self.debug = debug
        self.icon = cv2.imread(icon_path)
        self.toolbar_left = toolbar_left
        self.kwargs = kwargs
        self.fx = 1
        self.fy = 1
        self.resize = False
        self.reset_region()

    def match_icon(self):
        target = self.grab_screen()
        bias_left = 0 if not self.toolbar_left else 60
        target = target[:int(target.shape[0] / 3), bias_left:int(target.shape[1] / 3)]
        res = cv2.matchTemplate(target, self.icon, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.8:
            return None
        return max_loc

    def adjust_region(self, icon_loc: tuple[int, int]):
        # find the length of banner
        sc = self.grab_screen()[:, 0 if not self.toolbar_left else 60:]
        banner_color = sc[icon_loc[1] - 2, icon_loc[0] - 2]
        x = icon_loc[0] - 2
        while (sc[icon_loc[1] - 2, x] - banner_color).sum() == 0:
            x -= 1
        x_ed = icon_loc[0] + self.icon.shape[1] + 2
        while (sc[icon_loc[1] - 2, x_ed] - banner_color).sum() == 0:
            x_ed += 1
        width = x_ed - x - 1
        if width - 1280 > 2 or width - 1280 < -2:
            self.resize = True
        expected_height = int(width * 720 / 1280)
        self.fx = 1280 / width
        self.fy = 720 / expected_height
        self.set_region(icon_loc[1] + 30, x, width, expected_height)

    def set_region(self, top: int, left: int, width: int, height: int):
        with mss.mss() as sct:
            monitor = sct.monitors[self.display_screen + 1]
            # set capture_coord by adding the base coordinate of the screen
            self.capture_coord = {"top": monitor["top"] + top, "left": monitor["left"] + left,
                                  "width": width, "height": height}

    def reset_region(self):
        with mss.mss() as sct:
            # notably, sct.monitors will have num_screens + 1 elements, the first element is the whole screen
            monitor = sct.monitors[self.display_screen + 1]
            self.capture_coord = {"top": monitor["top"], "left": monitor["left"], "width": monitor["width"],
                                  "height": monitor["height"]}

    def save_image(self, img: np.ndarray, save_dir: str = None):
        # get number of images in the directory
        if save_dir is None:
            save_dir = self.save_dir
        # create dir if save_dir not exist
        os.makedirs(save_dir, exist_ok=True)
        num_images = len(os.listdir(save_dir))
        # save the image
        cv2.imwrite(os.path.join(self.save_dir, f'{num_images:04d}.png'), img)

    def grab_screen(self) -> np.ndarray:
        """
        Grab the screen and return the image based on self.capture_coord
        Note that the image is in BGR format
        """
        with mss.mss() as sct:
            image = sct.grab(self.capture_coord)
            image = np.array(image)[:, :, :-1]
            if self.resize:
                # print(f'size: {image.shape}')
                image = cv2.resize(image, (int(image.shape[1] * self.fx), int(image.shape[0] * self.fy)), interpolation=cv2.INTER_CUBIC)
                # print(f'size: {image.shape}')
            return image
