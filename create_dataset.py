import os
import yaml
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import random
import numpy as np

# Load the YAML file
with open('data.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# Load the font files
font_cn = TTFont("font/S3G_Display_cn.ttf")
font_en = TTFont("font/S3G_Display_en.ttf")


# Check if a character exists in a font file
def char_exists(font, unicode_val):
    for table in font["cmap"].tables:
        if table.isUnicode():
            if int(unicode_val, 16) in table.cmap:
                return True
    return False

# Create the output directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Loop through the YAML data
for item in data:
    char = item['content']
    index = item['index']
    unicode_val = item['unicode']

    # Check if the character exists in the font files
    if char_exists(font_cn, unicode_val):
        font_file = "font/S3G_Display_cn.ttf"
    elif char_exists(font_en, unicode_val):
        font_file = "font/S3G_Display_en.ttf"
    else:
        print(f"Character '{char}' not found in either font file.")
        continue

    # Generate and save the image
    for fidx, fs in enumerate([48, 64]):
        img = Image.new('RGB', (fs, fs), color='black')
        draw = ImageDraw.Draw(img)
        # random font size by fs - [12, 5, 15]
        fs_bias = random.randint(-12, 5)
        font = ImageFont.truetype(font_file, fs + fs_bias)
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # random bias x and y from -5 to 5
        x_bias = random.randint(-5, 5)
        y_bias = random.randint(-5, 5)
        x = (fs - w) / 2 + x_bias
        y = (fs - h) / 2 - bbox[1] + y_bias  # adjust the y-coordinate to center the text vertically
        draw.text((x, y), char, font=font, fill='white')
        img.save(f"dataset/{index:04d}-{fidx}.jpg")
        img_np = np.asarray(img).copy()
        img_np[img_np < 255*.95] = 0
        Image.fromarray(img_np).save(f"dataset/{index:04d}-{fidx}-1.jpg")

