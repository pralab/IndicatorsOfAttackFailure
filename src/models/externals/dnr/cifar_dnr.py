"""Pretrained network from https://github.com/aaron-xichen/pytorch-playground"""

import torch
from torch import nn, optim

from secml.ml.classifiers import CClassifierPyTorch


class Flatten(nn.Module):
    """Layer custom per reshape del tensore
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    p = 0.1
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Dropout(p)]
            p += 0.1
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False,
                                                  momentum=0.9),
                           nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def cifar10(lr=1e-2, momentum=0.9, weight_decay=1e-2, preprocess=None,
            softmax_outputs=False, random_state=None, epochs=75, gamma=0.1,
            batch_size=100, lr_schedule=(25, 50), n_channel=64):
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M',
           4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    model.features = nn.Sequential(*model.features, Flatten())
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, gamma)
    return CClassifierPyTorch(model=model, loss=loss, optimizer=optimizer,
                              optimizer_scheduler=scheduler, epochs=epochs,
                              input_shape=(3, 32, 32), preprocess=preprocess,
                              random_state=None, batch_size=batch_size,
                              softmax_outputs=softmax_outputs)
