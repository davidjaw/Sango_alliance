import mss
import numpy as np
import cv2
from typing import List, Tuple, Dict, Union, Callable, Optional
import os
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Normalize
import torch.nn.functional as F
from torch import Tensor
import yaml
import copy


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


def resize(img: torch.Tensor, post_resize: List[int] | Tuple[int] | None = None):
    flag = False
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        flag = True
    img = F.interpolate(img, size=post_resize, mode='nearest')
    if flag:
        img = img.squeeze(0)
    return img


def left_align_and_pad(img_size: List[int] | Tuple[int] | None = None, target_w: int = 150,
                       target_h: int = 18):
    if img_size is None:
        raise Exception("img_size must be specified.")

    def func(images: np.ndarray | List[np.ndarray]):
        if not isinstance(images, list):
            images = [images]
        imgs = []
        for img in images:
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            img_h = img.shape[0]
            if img_h < target_h:
                img = np.pad(img, ((0, target_h - img_h), (0, 0), (0, 0)), mode='constant')
            img_w = img.shape[1]
            if img_w < target_w:
                img = np.pad(img, ((2, 2), (3, target_w - img_w - 3), (0, 0)), mode='constant')
            imgs.append(img)

        # cv2.imshow('img', np.concatenate(imgs, axis=0).astype(np.uint8))
        # cv2.waitKey(1)
        batch_img = np.asarray(imgs, dtype=np.float32) / 255.
        batch_img = resize(torch.from_numpy(batch_img).permute(0, 3, 1, 2), img_size)
        batch_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(batch_img)

        return batch_img

    return func


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        r""" This is a custom transformer decoder layer that does not use `memory` for a Decoder-only design.
        This function also removed the `_mha_block` since it's attention over tgt and memory.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def forward(
            self,
            tgt: Tensor,
            tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            **kwargs
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        r""" This custom function only change the way how `self.layers` is initialized. """
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer] + [copy.deepcopy(decoder_layer) for _ in range(num_layers - 1)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Image2Sequence(nn.Module):
    def __init__(self, seq_len, d_model, inter_c=384):
        r""" This function transform an image into a sequence of patches by
        dividing them horizontally into `seq_len` pieces.
        [batch, c, h, w] -> [seq_len, batch, c * h * (w // seq_len)]
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.output_layer = nn.Sequential(
            nn.Linear(inter_c, d_model),
            nn.Hardswish()
        )
        self.pos_embedding = nn.Parameter(torch.empty(seq_len, 1, d_model))
        nn.init.uniform_(self.pos_embedding)

    def forward(self, images):
        batch_size, channels, height, width = images.shape
        if width % self.seq_len != 0:
            raise ValueError(f"Image width ({width}) must be divisible by seq_len ({self.seq_len})")

        patch_w = width // self.seq_len
        patch_h = height

        # Reshape and permute the dimensions to get [seq_len, batch_size, channels, patch_size, patch_size]
        images = images.view(batch_size, channels, patch_h, self.seq_len, patch_w)
        images = images.permute(3, 0, 1, 4, 2)

        # Flatten the patches and apply positional encoding
        patches = torch.reshape(images, (self.seq_len, batch_size, channels * patch_h * patch_w))
        patches = self.output_layer(patches)
        patches += self.pos_embedding

        return patches


class CustomOCRNet(nn.Module):
    def __init__(
            self,
            attention_heads=4,
            seq_len=6,
            class_num=4870,
            batch_size=1,
            **factory_kwargs
    ):
        super(CustomOCRNet, self).__init__()
        output_c = 512
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.features[:9]))
        self.mobilenet_post = mobilenet_v3_small(pretrained=True)
        self.mobilenet_post = nn.Sequential(*list(self.mobilenet_post.features[9:]))
        self.pre_seq_estimation = nn.Conv2d(576, 128, kernel_size=3, padding=1, stride=2)
        self.seq_length = seq_len
        self.attention_heads = attention_heads
        self.max_pool = nn.AdaptiveMaxPool1d(seq_len)
        self.seq_estimation_head = nn.Linear(768, seq_len, bias=False)
        self.merge_layer = nn.Sequential(
            nn.Conv2d(176, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
        )
        self.img_to_seq = Image2Sequence(seq_len=seq_len * 2, d_model=output_c)
        self.transformer = CustomTransformerDecoderLayer(d_model=output_c, nhead=attention_heads,
                                                         dim_feedforward=output_c // 2, norm_first=True,
                                                         **factory_kwargs)
        self.transformer_network = CustomTransformerDecoder(self.transformer, num_layers=2)
        # Regression head: output is a sequence of length 0 to 6, each element being a regression output for width
        self.regression_head = nn.Sequential(
            nn.Linear(output_c, 1),
            nn.Tanh()  # Ensure the output is between -1 and 1
        )
        self.classification_head = nn.Linear(output_c, class_num, bias=False)

    def forward(self, images):
        spatial_feature = self.mobilenet(images)
        semantic_feature = self.mobilenet_post(spatial_feature)
        semantic_feature = self.pre_seq_estimation(semantic_feature)

        semantic_skip_feature = F.interpolate(semantic_feature,
                                              size=(spatial_feature.shape[2], spatial_feature.shape[3]),
                                              mode='bilinear', align_corners=True)

        semantic_feature = torch.flatten(semantic_feature, start_dim=1)
        seq_logit = self.seq_estimation_head(semantic_feature)

        spatial_feature = torch.concat([spatial_feature, semantic_skip_feature], dim=1)
        spatial_feature = self.merge_layer(spatial_feature)
        seq_feature = self.img_to_seq(spatial_feature)
        feature = self.transformer_network(seq_feature)
        feature = self.max_pool(feature.permute(1, 2, 0))
        feature = feature.permute(0, 2, 1)

        # Get regression and classification outputs
        regression_outputs = self.regression_head(feature)
        classification_outputs = self.classification_head(feature)

        return regression_outputs, classification_outputs, seq_logit

    @staticmethod
    def decode_output(out_r, out_c, out_seq, data_file='data.yaml', data=None):
        if data is None:
            data = yaml.load(open(data_file, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        result = []
        batch_size = out_r.shape[0]
        for i in range(batch_size):
            r, c, seq = out_r[i], out_c[i], out_seq[i]
            c = c[:seq + 1]
            text = ''.join([data[x]['content'] for x in c.tolist()])
            result.append(text)
        return result


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
