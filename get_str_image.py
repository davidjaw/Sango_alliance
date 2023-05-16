import cv2
import yaml
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import numpy as np
from typing import List, Tuple
import random


class FontDrawer:
    def __init__(
            self,
            img_size: int = 20,
            font_size: int = 20,
            max_random_w: int = 5,
            max_str_len: int = 6,
            data_yaml_path='data.yaml',
    ):
        # Load the YAML file
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        # Load the font files
        self.font_cn = TTFont("font/S3G_Display_cn.ttf")
        self.font_en = TTFont("font/S3G_Display_en.ttf")
        self.img_size = img_size
        self.font_size = font_size
        self.max_random_w = max_random_w
        self.max_str_len = max_str_len

    def draw(self, target_str: str) -> Tuple[List[np.ndarray], List[int]]:
        # search data to find content == target_str
        target_idx = [-1 for _ in range(len(target_str))]
        for idx, item in enumerate(self.data):
            if item['content'] in target_str:
                char_idx = target_str.index(item['content'])
                target_idx[char_idx] = idx

        result = []
        indices = []
        for idx in target_idx:
            item = self.data[idx]
            char = item['content']
            index = item['index']
            unicode_val = item['unicode']

            font_file = self.get_font_file(unicode_val)
            if font_file is None:
                raise Exception(f"Character '{char}' not found in either font file.")

            # Create the image
            img = self.draw_char(img_size=20, font_size=20, target_font_file=font_file, target_char=char)
            img = np.asarray(img).copy()
            result.append(img)
            indices.append(index)
        return result, indices

    def get_font_file(self, unicode_val) -> None | str:
        # Check if the character exists in the font files
        if self.char_exists(self.font_cn, unicode_val):
            font_file = "font/S3G_Display_cn.ttf"
        elif self.char_exists(self.font_en, unicode_val):
            font_file = "font/S3G_Display_en.ttf"
        else:
            return None
        return font_file

    def draw_n(
            self,
            str_len: int,
            transform=None,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        random_sampled = random.sample(self.data, str_len)

        result = None
        indices = []
        spaces = []
        for item in random_sampled:
            char = item['content']
            index = item['index']
            unicode_val = item['unicode']

            font_file = self.get_font_file(unicode_val)
            if font_file is None:
                raise Exception(f"Character '{char}' not found in either font file.")

            # Create the image
            img = self.draw_char(img_size=self.img_size, font_size=self.font_size, target_font_file=font_file, target_char=char)
            img = np.asarray(img).copy()
            if transform is not None:
                img = transform(Image.fromarray(img))
                img = np.asarray(img)
            img_flat = np.max(img, axis=(0, -1))
            x_st = np.nonzero(img_flat)[0][0]
            x_ed = np.nonzero(img_flat)[0][-1]
            img = img[:, x_st:x_ed+1, :]
            random_w = random.randint(0, self.max_random_w)
            x_padd = np.zeros((img.shape[0], random_w, img.shape[2]), dtype=np.uint8)
            if result is None:
                result = np.concatenate((x_padd, img), axis=1)
            else:
                result = np.concatenate((result, x_padd, img), axis=1)
            spaces.append(img.shape[1] + random_w)
            indices.append(index)
        for i in range(self.max_str_len - str_len):
            indices.append(-1)
            spaces.append(-1)
        return result, indices, spaces

    @staticmethod
    def char_exists(font, unicode_val) -> bool:
        for table in font["cmap"].tables:
            if table.isUnicode():
                if int(unicode_val, 16) in table.cmap:
                    return True
        return False

    @staticmethod
    def draw_char(img_size, font_size, target_font_file, target_char) -> Image:
        text_img = Image.new('RGB', (img_size, img_size), color='black')
        draw = ImageDraw.Draw(text_img)
        font = ImageFont.truetype(target_font_file, font_size)
        bbox = draw.textbbox((0, 0), target_char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (font_size - w) / 2
        y = (font_size - h) / 2 - bbox[1]
        draw.text((x, y), target_char, font=font, fill='white')
        return text_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    drawer = FontDrawer()
    """ Test draw specific string """
    # result, indices = drawer.draw('測試一下A到Z會長怎樣0123456789')
    # final_img = np.concatenate(result, axis=1)
    # cv2.imshow('final', final_img)
    # cv2.waitKey(0)

    """ Test draw_n """
    # for _ in range(20):
    #     result, indices, spaces = drawer.draw_n(10)
    #     result_flat = np.max(result, axis=0, keepdims=True)
    #     result_flat[result_flat > 0] = 255
    #     final = cv2.vconcat([result, result_flat])
    #     final = cv2.resize(final, (final.shape[1] * 5, final.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    #     print(spaces)
    #     cv2.imshow('final', final)
    #     cv2.waitKey(0)

    """ check the distribution of the char's width """
    data = drawer.data
    widths = []
    for idx, d in enumerate(data):
        char = d['content']
        unicode_val = d['unicode']
        font_file = drawer.get_font_file(unicode_val)
        if font_file is None:
            raise Exception(f"Character '{char}' not found in either font file.")
        img = drawer.draw_char(img_size=20, font_size=20, target_font_file=font_file, target_char=char)
        img = np.asarray(img).copy()
        img_flat = np.max(img, axis=(0, -1))
        x_st = np.nonzero(img_flat)[0][0]
        x_ed = img.shape[1] - np.nonzero(img_flat[::-1])[0][0]
        widths.append(x_ed - x_st - 1)
    plt.hist(widths, bins=len(set(widths)))
    plt.show()
    print(set(widths))
    # {2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
