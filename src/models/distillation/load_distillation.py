import os

import torch
from robustbench.utils import download_gdrive
from secml.array import CArray
from secml.ml import CClassifierPyTorch
from torch import nn

MODEL_ID = '1YqJwAm6JgeUcWZmPsyNA_UeZMcw6FUP9'


class MNIST9Layer(nn.Module):
    def __init__(self):
        super(MNIST9Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 200)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.linear3(x)

        return x


def load_model():
    # T = 100 for this model
    model = MNIST9Layer()
    path = os.path.join(os.path.dirname(__file__), 'distilled_mnist_student.pt')
    if not os.path.exists(path):
        download_gdrive(MODEL_ID, path)
    state_dict = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    model = CClassifierPyTorch(model, input_shape=(1, 28, 28), pretrained=True,
                               pretrained_classes=CArray(list(range(10))), preprocess=None)
    return model
