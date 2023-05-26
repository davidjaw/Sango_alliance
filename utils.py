import mss
import numpy as np
import cv2
from typing import List, Tuple, Dict
import os
import torch
from torchvision import transforms
import math


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


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss

