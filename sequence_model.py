import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable, Tuple, List, Optional, Dict
import math
import copy


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, seq_len=1, batch_size=1, with_mhca=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.with_mhca = with_mhca
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
        if self.with_mhca:
            self.multihead_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                              **factory_kwargs)
            self.pre_cross_attention = nn.Linear(seq_len, d_model, **factory_kwargs)
            self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.dropout3 = Dropout(dropout)

        self.activation = activation
        self.index_vector = torch.eye(seq_len, **factory_kwargs)
        self.index_vector = self.index_vector.unsqueeze(0).repeat(batch_size, 1, 1)
        self.index_vector = self.index_vector.to(device)

    def _mhca_block(self, x: Tensor) -> Tensor:
        query = self.pre_cross_attention(self.index_vector)
        query = query.view(query.size(1), query.size(0), -1)
        x = self.multihead_cross_attn(query, x, x)[0]
        return self.dropout3(x)

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
            if self.with_mhca:
                x = x + self._mhca_block(self.norm3(x))
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            if self.with_mhca:
                x = self.norm3(x + self._mhca_block(x))
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

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
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
            # make sure that the width is divisible by seq_len, pad with zeros if necessary
            w = (width // self.seq_len + 1) * self.seq_len
            images = F.pad(images, (0, w - width), mode='constant', value=0)
            width = w

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
                                                         batch_size=batch_size, seq_len=seq_len * 2,
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


def main():
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 3
    model = CustomOCRNet(batch_size=batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    random_inputs = torch.randn(batch_size, 3, 48, 380).to(device)
    print("Input shape: {}".format(random_inputs.shape))
    for i in range(2):
        regression_outputs, classification_outputs, seq_len_logit = model(random_inputs)
        print("Regression output shape: {}".format(regression_outputs.shape))
        print("Classification output shape: {}".format(classification_outputs.shape))
        print("Sequence length logit shape: {}".format(seq_len_logit.shape))
    print('finished warmup. starting timer')

    st_time = time.time()
    for i in range(100):
        model(random_inputs)
    total_time = time.time() - st_time
    print(f'100 forward passes took {total_time} seconds, FPS={100 / total_time}')
    torch.save(model.state_dict(), 'weights/ocr-test.pth')


if __name__ == "__main__":
    main()
