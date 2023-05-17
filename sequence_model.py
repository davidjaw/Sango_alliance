import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from transformers import DistilBertModel
import torch.nn.functional as F


class CustomOCRNet(nn.Module):
    def __init__(
            self,
            attention_heads=4,
            seq_len=6,
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
        self.seq_estimation_head = nn.Linear(128 * 7, 5, bias=False)
        self.pre_distilbert = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=3, padding=0, stride=2),
            nn.Hardswish(),
            nn.Conv2d(128, 768, kernel_size=1, padding=1)
        )
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Regression head: output is a sequence of length 0 to 6, each element being a regression output for width
        self.regression_head = nn.Sequential(
            nn.Linear(self.distilbert.config.dim, 1),
            nn.Tanh()  # Ensure the output is between -1 and 1
        )
        self.classification_head = nn.Linear(self.distilbert.config.dim, 24 - 2 + 1, bias=False)

    def forward(self, images):
        features = self.mobilenet(images)
        img_features = self.mobilenet_post(features)
        img_features = self.pre_seq_estimation(img_features)
        img_features = torch.flatten(img_features, start_dim=1)
        seq_logit = self.seq_estimation_head(img_features)  # Estimate the sequence length
        seq_length = 1 + torch.argmax(seq_logit, dim=1)  # Get the estimated sequence length

        features = self.pre_distilbert(features)
        features = F.hardswish(features)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.transpose(1, 2)
        outputs = self.distilbert(inputs_embeds=features).last_hidden_state
        outputs = outputs.transpose(1, 2)  # swap dimensions for AdaptiveMaxPool1d
        outputs = self.max_pool(outputs)
        outputs = outputs.transpose(1, 2)  # swap dimensions back

        # Get regression and classification outputs
        regression_outputs = self.regression_head(outputs)
        classification_outputs = self.classification_head(outputs)

        return regression_outputs, classification_outputs, seq_logit


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomOCRNet()
    model.to(device)
    random_inputs = torch.randn(2, 3, 16*3, 130*3).to(device)
    regression_outputs, classification_outputs, seq_len_logit = model(random_inputs)
    print(regression_outputs.shape)
    print(classification_outputs.shape)
    print()


if __name__ == "__main__":
    main()
