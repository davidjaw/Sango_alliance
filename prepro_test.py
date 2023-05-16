from typing import List
import numpy as np
import cv2
import os


def preprocess_img(img: np.ndarray, color_cond=None, std_cond=None) -> List[np.ndarray]:
    img_std = np.std(img, axis=-1)
    if std_cond is None:
        img_std_cond = img_std < img_std.mean()
    else:
        img_std_cond = np.where(std_cond(img_std), True, False)
    if color_cond is None:
        img_color_cond = np.max(img, axis=-1) > 125
        mask = np.logical_and(img_std_cond, img_color_cond)
    else:
        img_color_cond = np.where(color_cond(img), True, False)
        mask = img_color_cond
    mask_gaussian = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), 0)

    mask = mask.astype(np.uint8) * 255
    mask_merged = np.concatenate((mask[None, :], mask_gaussian[None, :]), axis=0)
    mask_merged = np.max(mask_merged, axis=0)
    mask_merged_norm = mask_merged / 255.
    masked_color_img = img.astype(np.float32) * mask_merged_norm[:, :, None]
    masked_color_img = masked_color_img / np.max(masked_color_img) * 255
    masked_color_img = masked_color_img.astype(np.uint8)
    masked_color_img = np.max(masked_color_img, axis=-1)
    masked_color_img = np.stack([masked_color_img, masked_color_img, masked_color_img], axis=-1)
    return [masked_color_img, mask_merged, mask, mask_gaussian]


def get_patch(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def preprocess_name(img: np.ndarray, mask: np.ndarray):
    mask_flat = np.max(mask, axis=0)
    cv2.imshow('img', img)
    cv2.imshow('mask', cv2.vconcat([mask, mask_flat[None, ...]]))

    rand_int = np.random.randint(1e7)
    cv2.imwrite(f'tmp/name/{rand_int}.png', cv2.vconcat([mask, mask_flat[None, ...]]))
    cv2.waitKey(1)


def preprocess_name_patch(img: np.ndarray):
    name_patch_coord = [
        [483, 118, 119, 19],  # x, y, w, h
        [681, 118, 119, 19],
    ]
    img_processed, mask_f, mask_i, mask_g = preprocess_img(img)
    patch_ally = [get_patch(x, *name_patch_coord[0]) for x in [img_processed, mask_f, mask_i]]

    patch_enemy = [get_patch(x, *name_patch_coord[1]) for x in [img_processed, mask_f, mask_i]]
    color_cond = lambda x: np.logical_or(np.mean(x, axis=-1) > 125, np.argmax(x, axis=-1) == 0)
    img_enemy_processed, mask_f, mask_i, mask_g = preprocess_img(img, color_cond=color_cond)
    patch_enemy2 = [get_patch(x, *name_patch_coord[1]) for x in [img_enemy_processed, mask_f, mask_i]]
    if np.count_nonzero(patch_enemy[0]) < np.count_nonzero(patch_enemy2[0]):
        patch_enemy = patch_enemy2

    patch_ally_name = preprocess_name(patch_ally[0], patch_ally[2])


def main():
    full_img_dir = 'tmp/sc'
    out_img_dir = 'tmp/full-processed'
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    img_list = os.listdir(full_img_dir)
    for img_name in img_list:
        img = cv2.imread(os.path.join(full_img_dir, img_name))
        preprocess_name_patch(img)


if __name__ == '__main__':
    main()

