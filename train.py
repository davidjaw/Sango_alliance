import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Lambda, Compose
from torchvision import transforms, models, datasets
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
import yaml
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import dataloader


class RandomPixelDrop:
    def __init__(self, drop_ratio=0.5):
        self.drop_ratio = drop_ratio

    def __call__(self, img):
        mode = random.choice(["region", "random", "h_drop", "v_drop"])
        apply_random = random.random() < 0.9
        img_np = np.array(img)
        h, w, *_ = img_np.shape
        h_drop = int(h * self.drop_ratio)
        w_drop = int(w * self.drop_ratio)
        if not apply_random:
            return Image.fromarray(img_np)

        if mode == "region":
            region = random.choice(["topleft", "topright", "bottomleft", "bottomright", "center"])
            if region == "topleft":
                img_np[:h_drop, :w_drop] = 0
            elif region == "topright":
                img_np[:h_drop, w-w_drop:] = 0
            elif region == "bottomleft":
                img_np[h-h_drop:, :w_drop] = 0
            elif region == "bottomright":
                img_np[h-h_drop:, w-w_drop:] = 0
            elif region == "center":
                img_np[h//2 - h_drop//2 : h//2 + h_drop//2, w//2 - w_drop//2 : w//2 + w_drop//2] = 0
        elif mode == "random":
            mask = np.random.random((h, w)) < self.drop_ratio
            img_np[mask] = 0
        elif mode == "h_drop" or mode == "v_drop":
            idx = np.ones(img_np.shape[0 if mode == "h_drop" else 1], dtype=np.uint8)
            random_idx = np.random.random(idx.shape) < self.drop_ratio
            idx[random_idx] = 0
            if mode == "h_drop":
                img_np = img_np * idx[:, np.newaxis, np.newaxis]
            else:
                img_np = img_np * idx[np.newaxis, :, np.newaxis]
        return Image.fromarray(img_np)


def filter_misclassified(cm):
    misclassified_indices = np.argwhere(np.triu(cm, 1) + np.tril(cm, -1) > 0)
    unique_indices = np.unique(misclassified_indices)
    filtered_cm = cm[np.ix_(unique_indices, unique_indices)]
    return filtered_cm, unique_indices


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, display_threshold=.75, filter_misclass=True):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if filter_misclass:
        cm_filtered, unique_indices = filter_misclassified(cm)
        if len(unique_indices) > 0:
            cm = cm_filtered
            target_names = [target_names[i] for i in unique_indices]

    num_classes = len(target_names)
    fig_size = max(8, num_classes // 4)
    font_size = max(8, 72 // num_classes)

    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=font_size)
        plt.yticks(tick_marks, target_names, fontsize=font_size)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if display_threshold is not None and cm[i, j] < display_threshold:
            continue
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=font_size,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=font_size,
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)
    plt.savefig('confusion_matrix.png')
    plt.close()


def confusion_matrix(preds, labels, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def main():
    task_target = 'ocr'
    torch.manual_seed(9527)
    num_class = len(os.listdir('dataset')) // 4
    # read config from 'nn_config.yaml'
    config = yaml.load(open('nn_config.yaml', 'r'), Loader=yaml.FullLoader)
    default_config = config['default']
    task_config = config[task_target]
    task_config = {**default_config, **task_config}
    print(f'config: {task_config}')
    img_size = task_config['img_size']
    batch_size = task_config['batch_size']
    train_ratio = task_config['train_ratio']
    hs_ratio = task_config['hs_ratio']
    drop_ratio = task_config['drop_ratio']

    train_transforms = Compose([
        # Lambda(dataloader.binarize),
        RandomPixelDrop(drop_ratio=drop_ratio),
        Lambda(dataloader.random_shift),
        Lambda(dataloader.random_scale),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        # Lambda(dataloader.binarize),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = dataloader.TextRecognitionDataset('dataset', train_transforms)
    valid_dataset = dataloader.TextRecognitionDataset('dataset', val_transforms)
    train_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    # model = models.mobilenet_v3_small()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
    # st_dict = torch.load(f'weights/{task_target}.pth')
    # model.load_state_dict(st_dict)
    model = model.to(device)

    # Assuming your DataLoader returns one batch of images and labels
    images, labels = next(iter(train_loader))
    images = images.to(device)
    writer = SummaryWriter(f'runs/{task_target}')
    writer.add_graph(model, images)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    num_epochs = 201

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        epoch_preds = []
        epoch_labels = []

        last_train_samples = None
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            last_train_samples = inputs

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f'[{epoch}/{num_epochs}]Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Log the scalar values
        writer.add_scalar('training loss', epoch_loss, epoch)
        writer.add_scalar('accuracy/training', epoch_acc, epoch)

        # Log the histograms of weights
        for name, param in model.named_parameters():
            if 'weight' in name:
                writer.add_histogram('w/' + name, param.data.cpu().numpy(), epoch)

        # decay learning rate
        if epoch % 2 == 0 and epoch != 0:
            scheduler.step()

        # Validation and confusion matrix
        if epoch % 5 == 0:
            model.eval()
            last_val_samples = None
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                # for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    last_val_samples = inputs

            # Apply the inverse transformation to the last validation inputs
            last_train_samples = last_train_samples.cpu()
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            writer.add_histogram('img/train', torch.flatten(last_train_samples), epoch)
            last_train_samples = [inv_normalize(t) for t in last_train_samples]

            # Record the last validation inputs with a 5x5 grid
            grid = vutils.make_grid(last_train_samples[:25], nrow=5, normalize=True, scale_each=True)
            grid = F.interpolate(grid.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            writer.add_image('image/train', grid, epoch)
            last_val_samples = last_val_samples.cpu()
            writer.add_histogram('img/valid', torch.flatten(last_val_samples), epoch)
            last_val_samples = [inv_normalize(t) for t in last_val_samples]
            grid = vutils.make_grid(last_val_samples[:25], nrow=5, normalize=True, scale_each=True)
            grid = F.interpolate(grid.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            writer.add_image('image/valid', grid, epoch)

            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            # conf_matrix = confusion_matrix(val_preds, val_labels, num_classes=num_class)
            # plot_confusion_matrix(conf_matrix, target_names=[str(i) for i in range(num_class)])

            # # Log the confusion matrix as an image
            # confusion_image = Image.open('confusion_matrix.png')
            # confusion_image_tensor = transforms.ToTensor()(confusion_image)
            # writer.add_image('Confusion Matrix', confusion_image_tensor, epoch)

            accuracy = np.count_nonzero(val_preds == val_labels) / len(val_labels)
            writer.add_scalar('accuracy/validation', accuracy, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            model_weight_path = f'weights/{task_target}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_weight_path)

    writer.close()


if __name__ == '__main__':
    main()

