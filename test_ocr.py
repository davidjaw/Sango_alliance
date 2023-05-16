import cv2
import numpy as np
import torch
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as TF
from torch import nn
from PIL import Image
import os
import yaml


def name_patch_preprocess(patch: np.ndarray):
    patch_flat = np.max(patch, axis=(0, -1))
    xs = []
    x = 0
    max_w = 20
    while x < patch_flat.shape[0] and patch_flat[x:].max() != 0:
        x_st = np.nonzero(patch_flat[x:])[0][0] + x
        x_ed = np.argmin(patch_flat[x_st:]) + x_st
        if x_ed == x_st:
            x_ed = patch_flat.shape[0]
        w = x_ed - x_st
        if w > max_w:
            x_ed = x_st + max_w
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
        if 4 <= w < 10:
            # print(f'w not in range: {w}')
            nxt_w, prev_w = None, None
            nxt_x_st, nxt_x_ed, prev_x_st, prev_x_ed = None, None, None, None
            # cv2.destroyAllWindows()
            # cv2.imshow('img', patch)
            if idx < len(xs) - 1:
                nxt_x_st, nxt_x_ed = xs[idx + 1]
                nxt_w = int(nxt_x_ed) - int(nxt_x_st) + w
                # print(f'w_next: {nxt_w}')
                # cv2.imshow('merged_patch', patch[:, x_st:nxt_x_ed])
            if idx > 0:
                prev_x_st, prev_x_ed = xs[idx - 1]
                prev_w = int(prev_x_ed) - int(prev_x_st) + w
                # print(f'w_prev: {prev_w}')
                # cv2.imshow('prev_patch', patch[:, prev_x_st:x_ed])

            within_rage = lambda num: 13 < num <= 20
            if nxt_w is None and within_rage(prev_w):
                merged_w.pop()
                merged_w.append([prev_x_st, x_ed])
            elif prev_w is None and within_rage(nxt_w):
                skip_next_flag = True
                merged_w.append([x_st, nxt_x_ed])
            else:
                nxt_flag = within_rage(nxt_w)
                prev_flag = within_rage(prev_w)
                if nxt_flag and prev_flag:
                    flat_nxt = np.max(patch[:, x_st:nxt_x_ed], axis=(0, -1))
                    flat_prev = np.max(patch[:, prev_x_st:x_ed], axis=(0, -1))
                    if flat_nxt.shape[0] - np.count_nonzero(flat_nxt) < flat_prev.shape[0] - np.count_nonzero(flat_prev):
                        skip_next_flag = True
                        merged_w.append([x_st, nxt_x_ed])
                    else:
                        merged_w.pop()
                        merged_w.append([prev_x_st, x_ed])
                elif nxt_flag:
                    skip_next_flag = True
                    merged_w.append([x_st, nxt_x_ed])
                elif prev_flag:
                    merged_w.pop()
                    merged_w.append([prev_x_st, x_ed])
                else:
                    merged_w.append([x_st, x_ed])
            # cv2.waitKey(0)
        else:
            merged_w.append([x_st, x_ed])
            continue

    # if len(merged_w) != len(xs):
    #     p = patch.copy()
    #     for x_st, x_ed in merged_w:
    #         p[-3:, x_st:x_ed] = [0, 0, 255]
    #     p = cv2.vconcat([p, np.max(patch, axis=0, keepdims=True), patch])
    #     p = cv2.resize(p, (p.shape[1] * 3, p.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    #     cv2.destroyAllWindows()
    #     cv2.imshow('merged_patch', p)
    #     cv2.waitKey(0)

    chars = []
    for x_st, x_ed in merged_w:
        char = patch[:, x_st:x_ed]
        char = Image.fromarray(char)
        char = zero_pad_to_square(char)
        char = TF.resize(char, [96, 96])
        chars.append(np.asarray(char))
    return chars


def zero_pad_to_square(img: Image):
    w, h = img.size
    size = max(w, h)
    pad_w = size - w
    pad_h = size - h
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
    padding = [p + 4 for p in padding]
    return TF.pad(img, padding, fill=0, padding_mode='constant')


def main():
    task_target = 'ocr'
    target_dict = yaml.load(open('data.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    class_num = len(target_dict)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        normalize
    ])
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(normalize.mean, normalize.std)],
        std=[1 / s for s in normalize.std]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v3_small()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
    st_dict = torch.load(f'weights/{task_target}_epoch_200.pth')
    model.load_state_dict(st_dict)
    model = model.to(device)

    name_dir = os.listdir('tmp/name')
    for name_path in name_dir:
        print(name_path)
        img = cv2.imread(f'tmp/name/{name_path}')
        imgs = name_patch_preprocess(img)
        img_tensor = torch.from_numpy(np.asarray(imgs) / 255.)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = transform(img_tensor).float().to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
            pred_list = pred.cpu().numpy().tolist()
            pred_str = ''.join([target_dict[i]['content'] for i in pred_list])

            inv_input = inv_normalize(img_tensor)
            inv_input = inv_input.permute(0, 2, 3, 1)
            inv_input = inv_input.cpu().numpy()
            inv_input = np.clip(inv_input, 0, 1)
            inv_input = (inv_input * 255).astype(np.uint8)
            inv_input = [inv_input[x] for x in range(inv_input.shape[0])]
            cv2.imshow('inv_input', cv2.vconcat(inv_input))
        cv2.imshow('img', img)
        print(pred_str)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
