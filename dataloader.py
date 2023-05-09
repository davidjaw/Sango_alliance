import os
import random
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda, Compose
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from torchvision import transforms


class TextRecognitionDataset(Dataset):
    def __init__(self, data_dir, augmentations=None):
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        label = int(self.image_files[idx].split('-')[0])

        if self.augmentations:
            image = self.augmentations(image)

        return image, label


def binarize(img, threshold=0.3):
    img = np.array(img)
    img_mask = img > threshold * 255
    img = (~img_mask).astype(np.uint8)
    return Image.fromarray(img)


def random_scale(img, scale_range=(0.7, 1.05)):
    choice = random.randint(0, 1)
    if choice == 0:
        return img
    scale = random.uniform(*scale_range)
    new_size = (int(img.width * scale), int(img.height * scale))
    img = resize(img, new_size)
    return img


def random_shift(img, shift_range=0.2):
    choice = random.randint(0, 1)
    if choice == 0:
        return img
    w, h = img.size
    pad_w, pad_h = int(shift_range * w), int(shift_range * h)
    img = ImageOps.expand(img, border=(pad_w, pad_h), fill=0)

    shift_x = int(random.uniform(-shift_range, shift_range) * w)
    shift_y = int(random.uniform(-shift_range, shift_range) * h)
    shifted_img = img.transform((w + 2 * pad_w, h + 2 * pad_h), Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))

    shifted_img = shifted_img.crop((pad_w, pad_h, pad_w + w, pad_h + h))
    return shifted_img


def random_drop(img):
    img_arr = np.array(img)
    choice = random.randint(0, 3)

    if choice == 0:  # Global drop
        mask = np.random.rand(*img_arr.shape) > 0.5
        img_arr[mask] = 0
    elif choice == 1:  # Center drop
        h, w = img_arr.shape
        mask = np.ones((h, w), dtype=bool)
        ch, cw = (int(0.8 * h), int(0.8 * w))
        top, left = (h - ch) // 2, (w - cw) // 2
        center_mask = np.random.rand(ch, cw) > 0.7
        mask[top:top + ch, left:left + cw] = center_mask
        img_arr[mask] = 0
    elif choice == 2:  # Column drop
        mask = np.random.rand(img_arr.shape[1]) > 0.7
        img_arr[:, mask] = 0
    else:  # Row drop
        mask = np.random.rand(img_arr.shape[0]) > 0.7
        img_arr[mask, :] = 0

    return Image.fromarray(img_arr)


if __name__ == '__main__':
    from torchvision.utils import make_grid
    augmentations = Compose([
        Lambda(binarize),
        Lambda(random_shift),
        Lambda(random_drop),
        Lambda(random_scale),
        transforms.Resize((164, 164)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = TextRecognitionDataset('dataset', augmentations)
    valid_dataset = TextRecognitionDataset('dataset', Lambda(binarize))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    for i, (images, labels) in enumerate(train_loader):
        # make grid and display
        grid = make_grid(images * 255, nrow=8, padding=2)
        img = to_pil_image(grid)
        img.show()
        print()



