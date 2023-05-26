import cv2
import numpy as np
from typing import List, Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models, datasets
from torchvision.transforms import Compose, ToTensor, RandomAffine, Resize, Normalize
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import itertools
import matplotlib.pyplot as plt
import random
import yaml
import torchvision
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau
from data_loader import CustomOCRDataset, prepare_data_transform
from sequence_model import CustomOCRNet
from dadaptation import DAdaptAdam, DAdaptLion, DAdaptSGD
from utils import Adafactor


def main():
    torch.manual_seed(9527)
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    data = yaml.load(open('data.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)

    nn_config = config['nn_config']
    img_size = nn_config['img_size']
    batch_size = nn_config['batch_size']
    train_size = 3e4
    valid_size = 1e3

    train_transform = prepare_data_transform()
    valid_transform = prepare_data_transform(mode='test')
    train_dataset = CustomOCRDataset(train_size, transform=train_transform, post_resize=img_size)
    valid_dataset = CustomOCRDataset(valid_size, transform=valid_transform, post_resize=img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomOCRNet(class_num=len(data), batch_size=batch_size,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
    # st_dict = torch.load(f'weights/ocr.pth')
    # model.load_state_dict(st_dict)
    model.to(device)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    writer = SummaryWriter(f'runs/Adafactor')
    writer.add_graph(model, images)

    loss_c = nn.CrossEntropyLoss()
    loss_r = nn.SmoothL1Loss()
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = DAdaptAdam(model.parameters(), lr=1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    # optimizer = DAdaptSGD(model.parameters(), lr=1.)
    optimizer = Adafactor(model.parameters(), warmup_init=True, relative_step=True)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    num_epochs = 156

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_sample = 0
        epoch_loss_r, epoch_loss_c, epoch_loss_s = 0, 0, 0

        last_train_samples = None
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            text_width = labels[0].to(device)
            text_idx = labels[1].to(device)
            label_mask = labels[2].to(device).bool()
            label_seq_len = torch.sum(label_mask, dim=-1, dtype=torch.int64) - 1
            last_train_samples = inputs

            optimizer.zero_grad()

            output_r, output_c_logit, output_seq = model(inputs)
            loss_regression = loss_r(output_r[label_mask].squeeze(-1), text_width[label_mask])
            loss_classification = loss_c(output_c_logit[label_mask], text_idx[label_mask])
            loss_sequence = loss_c(output_seq, label_seq_len)
            loss = loss_regression + loss_classification * 10 + loss_sequence

            epoch_loss_c += loss_classification.detach().cpu().item()
            epoch_loss_r += loss_regression.detach().cpu().item()
            epoch_loss_s += loss_sequence.detach().cpu().item()

            loss.backward()
            optimizer.step()

            pred = torch.argmax(output_c_logit[label_mask].squeeze(-1), dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred == text_idx[label_mask].data)
            running_sample += torch.count_nonzero(label_mask).item()

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / running_sample
        print(f'[{epoch}/{num_epochs}]Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Log the scalar values
        writer.add_scalar('loss/total', epoch_loss, epoch)
        writer.add_scalar('loss/classification', epoch_loss_c / train_size, epoch)
        writer.add_scalar('loss/regression', epoch_loss_r / train_size, epoch)
        writer.add_scalar('loss/sequence', epoch_loss_s / train_size, epoch)
        writer.add_scalar('accuracy/training', epoch_acc, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Log the histograms of weights
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                writer.add_histogram(name, param.data.cpu().numpy(), epoch)

        # # decay learning rate
        # if epoch % 10 == 0 and epoch != 0:
        #     scheduler.step(epoch_loss)

        # Validation and confusion matrix
        if epoch % 5 == 0:
            model.eval()
            last_valid_samples = None
            val_preds = []
            val_labels = []
            val_seq_labels = []
            val_seq_preds = []
            val_sample = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    last_valid_samples = inputs
                    label_mask = labels[2].to(device).bool()
                    text_idx = labels[1].to(device)
                    label_seq_len = torch.sum(label_mask, dim=-1, dtype=torch.int64) - 1
                    output_r, output_c_logit, output_seq = model(inputs)

                    val_preds.extend(torch.argmax(output_c_logit, -1)[label_mask].cpu().numpy())
                    val_labels.extend(text_idx[label_mask].cpu().numpy())
                    val_seq_preds.extend(output_seq.cpu().numpy())
                    val_seq_labels.extend(label_seq_len.cpu().numpy())
                    val_sample += torch.count_nonzero(label_mask).item()

            # Apply the inverse transformation to the last validation inputs
            last_train_samples = last_train_samples.cpu()
            last_valid_samples = last_valid_samples.cpu()
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            last_train_samples = [inv_normalize(t) for t in last_train_samples]
            last_valid_samples = [inv_normalize(t) for t in last_valid_samples]

            # Record the last validation inputs with a 5x5 grid
            grid = vutils.make_grid(last_train_samples[:25], nrow=5, normalize=True, scale_each=True)
            grid = F.interpolate(grid.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            writer.add_image('images/train', grid, epoch)
            grid = vutils.make_grid(last_valid_samples[:25], nrow=5, normalize=True, scale_each=True)
            grid = F.interpolate(grid.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            writer.add_image('images/valid', grid, epoch)

            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)

            accuracy = np.count_nonzero(val_preds == val_labels) / val_sample
            writer.add_scalar('accuracy/valid_class', accuracy, epoch)
            accuracy = np.count_nonzero(np.argmax(val_seq_preds, axis=1) == val_seq_labels) / len(val_seq_labels)
            writer.add_scalar('accuracy/valid_seq_len', accuracy, epoch)

            model_weight_path = f'weights/ocr_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_weight_path)

    writer.close()


if __name__ == '__main__':
    main()
