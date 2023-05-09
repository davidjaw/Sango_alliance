import cv2
import numpy as np
import torch
from torchvision import transforms, models, datasets
from torch import nn
from PIL import Image
import os
import yaml


def name_patch_preprocess(patch: np.ndarray):
    patch_flat = np.max(patch, axis=(0, -1))
    xs = []
    x = 0
    while x < patch_flat.shape[0] and patch_flat[x:].max() != 0:
        x_st = np.nonzero(patch_flat[x:])[0][0] + x
        x_ed = np.argmin(patch_flat[x_st:]) + x_st
        if x_ed == x_st:
            x_ed = patch_flat.shape[0]
        w = x_ed - x_st
        if w > 1:
            xs.append((int(x_st), int(x_ed)))
        x = x_ed

    merged_w = []
    skip_next_flag = False
    for idx, x_coord in enumerate(xs):
        if skip_next_flag:
            skip_next_flag = False
            continue
        x_st, x_ed = x_coord
        w = int(x_ed) - int(x_st)
        if 5 < w < 10:
            print(f'w not in range: {w}')
            # cv2.destroyAllWindows()
            # cv2.imshow('img', patch)
            if idx < len(xs) - 1:
                nxt_x_st, nxt_x_ed = xs[idx + 1]
                nxt_w = int(nxt_x_ed) - int(nxt_x_st) + w
                print(f'w_next: {nxt_w}')
                # cv2.imshow('merged_patch', patch[:, x_st:nxt_x_ed])
                if 13 < nxt_w < 20:
                    skip_next_flag = True
                    merged_w.append([x_st, nxt_x_ed])
            if idx > 0:
                prev_x_st, prev_x_ed = xs[idx - 1]
                prev_w = int(prev_x_ed) - int(prev_x_st) + w
                print(f'w_prev: {prev_w}')
                # cv2.imshow('prev_patch', patch[:, prev_x_st:x_ed])
                if 13 < prev_w < 20:
                    merged_w.pop()
                    merged_w.append([prev_x_st, x_ed])
            # cv2.waitKey(0)
        else:
            merged_w.append([x_st, x_ed])
            continue

    if len(merged_w) != len(xs):
        p = patch.copy()
        for x_st, x_ed in merged_w:
            cv2.rectangle(p, (x_st, 0), (x_ed, p.shape[0]), (0, 0, 255), 1)
        p = cv2.vconcat([p, patch])
        cv2.destroyAllWindows()
        cv2.imshow('merged_patch', p)
        cv2.waitKey(0)


def main():
    task_target = 'ocr'
    target_dict = yaml.load(open('data.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    class_num = len(target_dict)
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v3_small()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
    st_dict = torch.load(f'weights/{task_target}.pth')
    model.load_state_dict(st_dict)
    model = model.to(device)

    name_dir = os.listdir('tmp/name')
    for name_path in name_dir:
        print(name_path)
        img = cv2.imread(f'tmp/name/{name_path}')
        img = name_patch_preprocess(img)


    print()


if __name__ == '__main__':
    main()
