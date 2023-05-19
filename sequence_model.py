import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F
import math


def positional_encoding(seq_len, d_model, device):
    PE = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE.unsqueeze(0)


class AdaptiveMaskAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            seq_len: int,
            batch_size: int = 1,
    ):
        super(AdaptiveMaskAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.act = nn.Hardswish()
        self.query = nn.Sequential(
            nn.Linear(seq_len, in_channels // 4),
            self.act,
            nn.Linear(in_channels // 4, out_channels // 4),
        )
        self.key = nn.Linear(in_channels, out_channels // 4 * seq_len)
        self.value = nn.Linear(in_channels, out_channels // 4 * seq_len)
        self.pre_out = nn.Linear(out_channels // 4, out_channels)

    def to_seq(self, x):
        # x is [batch, c], we need to convert it to [batch, seq_len, c // seq_len]
        assert self.batch_size % self.seq_len == 0
        x = x.view(self.batch_size, self.seq_len, -1)
        return x

    def forward(self, x):
        index_v = torch.eye(self.seq_len, device=x.device)
        index_v = index_v.unsqueeze(0).repeat(x.shape[0], 1, 1)
        index_v = index_v.view(-1, self.seq_len)
        query = self.query(index_v)
        query = query.view(x.shape[0], self.seq_len, -1)
        key = self.to_seq(self.key(x))
        value = self.to_seq(self.value(x))
        attention = query @ key.transpose(-1, -2)
        attention = attention / (self.out_channels // 4) ** 0.5
        attention = F.softmax(attention, dim=-1)
        out = attention @ value
        out = self.pre_out(out)
        out = self.act(out)
        return out


class SpatialMerge(nn.Module):
    def __init__(
            self,
            in_c_semantic: int,
            in_c_spatial: int,
            out_channels: int,
            seq_len: int,
            batch_size: int = 1,
            activation: nn.Module = nn.Hardswish(),
    ):
        super(SpatialMerge, self).__init__()
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_c_spatial, out_channels // 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels // 8),
            activation,
            nn.Flatten(),
            nn.Linear(23 * out_channels // 8, out_channels // 4),
            activation,
        )
        self.semantic_branch = nn.Sequential(
            nn.Linear(in_c_semantic, out_channels // 4),
            activation,
        )
        self.merge_layer = nn.Sequential(
            nn.Linear(out_channels // 2, out_channels),
            activation,
        )
        # the input of seq_attention is merged features
        self.seq_attention = AdaptiveMaskAttention(out_channels, out_channels, seq_len, batch_size=batch_size)
        self.out_channels = out_channels

    def forward(self, spatial_features, semantic_features):
        spatial_features = self.spatial_branch(spatial_features)
        semantic_features = self.semantic_branch(semantic_features)
        features = torch.cat([spatial_features, semantic_features], dim=1)
        features = self.merge_layer(features)
        features = self.seq_attention(features)
        return features


class CustomOCRNet(nn.Module):
    def __init__(
            self,
            attention_heads=4,
            seq_len=6,
            class_num=4870,
            batch_size=1,
    ):
        super(CustomOCRNet, self).__init__()
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.features[:9]))
        self.mobilenet_post = mobilenet_v3_small(pretrained=True)
        self.mobilenet_post = nn.Sequential(*list(self.mobilenet_post.features[9:]))
        self.pre_seq_estimation = nn.Conv2d(576, 128, kernel_size=3, padding=1, stride=2)
        self.seq_length = seq_len
        self.attention_heads = attention_heads
        self.max_pool = nn.AdaptiveMaxPool1d(seq_len)
        self.seq_estimation_head = nn.Linear(896, seq_len, bias=False)
        self.merge_layer = SpatialMerge(in_c_spatial=48, in_c_semantic=896, out_channels=768, seq_len=seq_len, batch_size=batch_size)
        # Regression head: output is a sequence of length 0 to 6, each element being a regression output for width
        self.regression_head = nn.Sequential(
            nn.Linear(768, 1),
            nn.Tanh()  # Ensure the output is between -1 and 1
        )
        self.classification_head = nn.Linear(768, class_num, bias=False)

    def forward(self, images):
        features = self.mobilenet(images)
        img_features = self.mobilenet_post(features)
        img_features = self.pre_seq_estimation(img_features)
        img_features = torch.flatten(img_features, start_dim=1)
        seq_logit = self.seq_estimation_head(img_features)

        features = self.merge_layer(features, img_features)
        # outputs = self.distilbert(inputs_embeds=features).last_hidden_state

        # Get regression and classification outputs
        regression_outputs = self.regression_head(features)
        classification_outputs = self.classification_head(features)

        return regression_outputs, classification_outputs, seq_logit


def main():
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomOCRNet()
    model.to(device)
    random_inputs = torch.randn(2, 3, 16*3, 130*3).to(device)
    print("Input shape: {}".format(random_inputs.shape))
    for i in range(2):
        regression_outputs, classification_outputs, seq_len_logit = model(random_inputs)
    print('finished warmup. starting timer')

    st_time = time.time()
    for i in range(100):
        model(random_inputs)
    total_time = time.time() - st_time
    print(f'100 forward passes took {total_time} seconds, FPS={100/total_time}')


if __name__ == "__main__":
    main()
