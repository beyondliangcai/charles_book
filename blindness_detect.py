# Libraries

import torch
import os
import gc
import sys
import cv2
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torchvision.models.resnet import ResNet, Bottleneck
import torch.optim as optim
from torch import nn


img_size = 224
batch_size = 32

# Functions

def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

    NOTE: This was used to generate the pre-processed dataset
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img):
    """
    Create circular crop around image centre

    NOTE: This was used to generate the pre-processed dataset
    """

    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img


class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(
            '../input/aptospreprocessed224x/train_images_processed_224x/train_images_processed_224x/',
            self.data.loc[idx, 'id_code'] + '.png.png')  # typo
        im = cv2.imread(img_name)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return {'image': im, 'labels': label}


class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images',
                                self.data.loc[idx, 'id_code'] + '.png')
        im = cv2.imread(img_name)
        im = circle_crop(im)
        im = cv2.resize(im, (img_size, img_size))
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return {'image': im}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.logsoftmax(out)

# processing
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform_train = Compose([
    ShiftScaleRotate(
        shift_limit=0.08,
        scale_limit=0.08,
        rotate_limit=365,
        p=1.0),
    RandomBrightnessContrast(p=1.0),
    ToTensor()
])
train_dataset = RetinopathyDatasetTrain(csv_file='../input/aptos2019-blindness-detection/train.csv',
                                        transform=transform_train)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/test.csv',
                                      transform=transform_train)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
