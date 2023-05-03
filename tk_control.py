from typing import List, Tuple
import os
import numpy as np
import cv2
import tkinter as tk
from utils import ScreenCap, NeuralNetworks
import keyboard
from tabulate import tabulate
from cnocr import CnOcr
import csv


def pad_img(img_to_pad, max_w, max_h):
    flag = False
    pad_bottom, pad_left = 0, 0
    if img_to_pad.shape[0] < max_h:
        pad_bottom = max_h - img_to_pad.shape[0]
        flag = True
    if img_to_pad.shape[1] < max_w:
        pad_left = max_w - img_to_pad.shape[1]
        flag = True
    if flag:
        # center padding for img
        padding = [(pad_bottom // 2, pad_bottom - pad_bottom // 2),
                   (pad_left // 2, pad_left - pad_left // 2)]
        if len(img_to_pad.shape) == 3:
            padding.append((0, 0))
        img_to_pad = np.pad(img_to_pad, padding, 'constant', constant_values=0)
    return img_to_pad[:max_h, :max_w]


class TkController(object):
    def __init__(
            self,
            tk_root: tk.Tk,
            screen_cap: ScreenCap,
            ocr: CnOcr | None,
            debug: bool = False,
            **kwargs,
    ):
        self.ocr = ocr
        self.tk_root = tk_root
        self.screen_cap = screen_cap
        self.debug = debug
        self.kwargs = kwargs
        # create widgets
        self.info_label = None
        self.detect_button = None
        self.create_widgets()
        self.data: List[List[str | int]] = []
        self.tk_root.protocol("WM_DELETE_WINDOW", self.on_leaving)

    def create_widgets(self):
        # create button
        self.detect_button = tk.Button(self.tk_root, text="Detect", command=self.detect_button_pressed, height=2)
        self.detect_button.pack(fill='x', padx=10, expand=True)
        # create label
        self.info_label = tk.Label(self.tk_root, text="")
        self.info_label.pack()
        # bind keyboard listener
        keyboard.add_hotkey('space', self.detect_button_pressed)

    def on_leaving(self):
        out_fn = self.kwargs["out_file"]
        print(f'Leaving program... saving data to {out_fn}')
        headers = ['ID', '貢獻', '戰功', '捐獻', '勢力值']
        with open(out_fn, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.data)
        self.tk_root.destroy()

    def detect_button_pressed(self):
        img = self.screen_cap.grab_screen()
        out_dir = 'tmp/sc'
        idx = 0
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            idx = len(os.listdir(out_dir))
        cv2.imwrite(f'{out_dir}/sc-{idx}.png', img)
        data = detect_user_block(img, reader=self.ocr, **self.kwargs)
        self.data.extend(data)

    def mainloop(self):
        self.tk_root.geometry("300x100")
        self.tk_root.mainloop()


def detect_user_block(img: np.ndarray, reader: CnOcr | None = None, model: NeuralNetworks | None = None, **_)\
        -> List[List[str | int]]:
    img = img[170:615, 30:840]
    idx = 0
    out_dir = 'tmp/sc/full'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        idx = len(os.listdir(out_dir))
    cv2.imwrite(f'{out_dir}/sc-{idx}.png', img)
    random_int = np.random.randint(0, 1e7)

    img_light = np.where(np.max(img, axis=-1) > 255 / 2.2, 255, 0).astype(np.uint8)
    img_light = np.stack([img_light] * 3, axis=-1)
    cv2.imwrite(f'{out_dir}/sc-{idx}-l.png', img_light)
    # cv2.imshow('img_light', img_light)
    # cv2.waitKey(0)

    # find y axis
    img_light_flat = np.max(img_light, axis=(1, 2))
    y_sections = []
    min_height = 10
    y = 0
    while y < img_light_flat.shape[0] - min_height and img_light_flat[y:].max() > 0:
        y_st = np.nonzero(img_light_flat[y:] > 0)[0][0]
        y_ed = np.argmin(img_light_flat[y + y_st:]) + y_st
        if y_ed - y_st > min_height:
            # divide into 5 part of x-axis image
            x_sections = np.array_split(img_light[y + y_st:y + y_ed], 5, axis=1)
            if x_sections[1].max() == 0:
                y += y_ed
                continue
            for idx, xs in enumerate(x_sections):
                xs_flat = np.max(xs, axis=0)
                xs_st = np.nonzero(xs_flat)[0][0]
                xs_ed = xs_flat.shape[0] - np.nonzero(xs_flat[::-1])[0][0]
                x_sections[idx] = xs[:, xs_st:xs_ed]
            y_sections.append(x_sections)
        else:
            y += y_ed
            continue
        y += y_ed

    user_data = [[] for _ in range(len(y_sections))]

    name_patch = []
    digit_patch = []
    for i, y_section in enumerate(y_sections):
        name_patch.append(y_section[0])
        digit_patch.extend(y_section[1:])
    # predict name
    # name_patch_np = np.asarray(name_patch)
    name_pred = reader.ocr_for_single_lines(name_patch)
    name_pred = [n['text'] for n in name_pred]
    for idx, name in enumerate(name_pred):
        # cv2.imshow(f'name-{idx}', name_patch[idx])
        user_data[idx].append(name)
    # cv2.waitKey(0)

    # get each digit and predict
    digit_length = []
    digits = []
    for d in digit_patch:
        d_flat = np.max(d, axis=(0, 2))
        cnt = 0
        x = 0
        while x < d_flat.shape[0] and np.max(d_flat[x:]) > 0:
            debug_d = d.copy()
            debug_d[:, x] = [0, 0, 255]
            # cv2.imshow('img', debug_d)
            # cv2.waitKey(0)
            x_st = np.nonzero(d_flat[x:])[0][0]
            x_ed = np.argmin(d_flat[x_st + x:]) + x_st
            if x_ed == x_st:
                x_ed = d_flat.shape[0] - x
            if x_ed - x_st > 1:
                digit = pad_img(d[:, x + x_st:x + x_ed], 10, 16)
                if x_ed - x_st >= 5 or (x_ed - x_st < 5 and np.count_nonzero(digit) > 10):
                    digits.append(digit)
                    cnt += 1
                else:
                    x += x_ed
                    continue
            x += x_ed
        digit_length.append(cnt)
        # print(f'digit length: {cnt}')
        # cv2.imshow('d', d)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    digits_np = np.asarray(digits)
    digits_pred = model.predict(digits_np).astype(str)
    cnt = 0
    for idx, u_data in enumerate(user_data):
        for i in range(4):
            d_len = digit_length[idx * 4 + i]
            num = int(''.join(digits_pred[cnt:cnt + d_len]))
            u_data.append(f'{num:,}')
            cnt += d_len
    # for idx, digit in enumerate(digits_pred):
    #     cv2.imwrite(f'tmp/digit/{digit}-{random_int}.png', digits_np[idx])
    table_format = "grid"
    print(tabulate(user_data, headers=['ID', '貢獻', '戰功', '捐獻', '勢力值'], tablefmt=table_format))
    return user_data


if __name__ == '__main__':
    import yaml
    reader = CnOcr(rec_model_name='chinese_cht_PP-OCRv3')
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    nn = NeuralNetworks(
        num_class=10,
        weights_path='digit.pth',
        config=config['nn_config']
    )
    imgs = os.listdir('tmp/sc')
    for img in imgs:
        img = cv2.imread(f'tmp/sc/{img}')
        detect_user_block(img, reader, nn)

