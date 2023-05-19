import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomAffine, Resize, RandomApply, Normalize
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import cv2
import numpy as np
import yaml
import torch.nn.functional as F
from get_str_image import FontDrawer


def prepare_data_transform(mode='train'):
    if mode == 'train':
        transform = Compose([
            transforms.RandomApply([
                RandomAffine(degrees=(-5, 5), scale=(0.95, 1.1), translate=(0.1, 0.1)),
            ], p=.8),
            ThresholdAugmentation(p=.5, min_threshold=100, max_threshold=130, p_g=.3),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2),
            ], p=.9),
        ])
    else:
        transform = Compose([
        ])
    return transform


class ThresholdAugmentation(object):
    def __init__(self, p, min_threshold, max_threshold, p_g):
        self.p = p
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.p_g = p_g

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_array = np.array(img)
        if random.random() <= self.p_g:
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        else:
            threshold = random.uniform(self.min_threshold, self.max_threshold)
            img_array = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)[1]
        img_augmented = transforms.ToPILImage()(img_array)
        return img_augmented


class CustomOCRDataset(Dataset):
    def __init__(self, num_samples, transform=None, post_resize=None):
        """ The transform is applied at the drawing stage, and post_transform is applied after the drawing stage. """
        drawer = FontDrawer()
        self.drawer = drawer
        self.num_samples = num_samples
        self.target_w = 150
        self.transform = transform
        self.post_resize = post_resize
        self.scale_factor = 1 if post_resize is None else post_resize[1] / self.target_w
        print('scale_factor', self.scale_factor)

    def resize(self, img: torch.Tensor):
        if self.post_resize is None:
            return img
        return F.interpolate(img.unsqueeze(0), size=self.post_resize, mode='nearest').squeeze(0)

    def width_denormalizer(self, widths):
        print(f'widths: {widths}, scale_factor: {self.scale_factor}')
        w = (widths + 1) * (19 + self.drawer.max_random_w - 2) * self.scale_factor / 2 + 2 * self.scale_factor
        print('w', w)
        return int(w)

    def width_normalizer(self, widths):
        return ((widths - 2) / (19 + self.drawer.max_random_w - 2)) * 2 - 1

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        word_length = torch.randint(1, 7, (1,)).item()
        image, text_ids, widths, seq_len = self.drawer.draw_n(word_length, transform=self.transform)
        image = ToTensor()(image)
        # if width of image < self.target_w, pad zero to the right
        if image.shape[2] < self.target_w:
            image = torch.nn.functional.pad(image, (0, self.target_w - image.shape[2], 0, 0), value=0)
        if self.post_resize is not None:
            image = self.resize(image)
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        widths = self.width_normalizer(torch.tensor(widths, dtype=torch.float32))
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)
        return image, (widths, text_ids, seq_len)


def test_dataloader():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_transform = prepare_data_transform()
    valid_transform = prepare_data_transform(mode='test')
    # train_dataset = CustomOCRDataset(60, transform=train_transform)
    valid_dataset = CustomOCRDataset(60, transform=valid_transform, post_resize=(16*3, 130*3))

    # Create the PyTorch DataLoader
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=30, shuffle=True, num_workers=4)
    w = set()
    for dataloader in [valid_dataloader]:
        for images, labels in dataloader:
            labels = [label.to(device) for label in labels]
            # make grid to show images, show row=6
            images = torch.permute(images, (0, 2, 3, 1))
            images_np = images.numpy()
            wx, text_id, seq_len = labels
            for i in range(images_np.shape[0]):
                img = images_np[i]
                x = 0
                x_widths = wx[i]
                for x_idx in range(valid_dataset.drawer.max_str_len):
                    x_width = valid_dataset.width_denormalizer(x_widths[x_idx])
                    if x_width > 0:
                        x += x_width
                        img[:, x, :] = [0, 0, 1]
                        w.add(x_width)
                # img_flat = np.max(img.mean(axis=-1), axis=0)
                # x_st = np.nonzero(img_flat)[0][0]
                # x_ed = img_flat.shape[0] - np.nonzero(img_flat[::-1])[0][0]
                # w.add(x_ed - x_st - 1)
            # show images
            images_np = np.transpose(images_np, (0, 3, 1, 2))
            img_grid = torchvision.utils.make_grid(torch.tensor(images_np), nrow=5)
            img_grid = img_grid.numpy().transpose((1, 2, 0))
            # img_grid = cv2.resize(img_grid, (img_grid.shape[1] * 3, img_grid.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('images', img_grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # import matplotlib.pyplot as plt
    # plt.hist(w, bins=range(min(w), max(w) + 1, 1), align='left')
    # plt.xlabel('value')
    # plt.ylabel('count')
    # plt.show()
    ws = sorted(w)
    print(ws)


def main():
    test_dataloader()


if __name__ == '__main__':
    main()

