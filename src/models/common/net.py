from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dim=None):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.dim = dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward_concrete(self, x):
        return self.conv(x)

    def forward_abstract(self, x):
        return x.conv2d(self.conv.weight, self.conv.bias, self.stride, self.conv.padding, self.dilation,
                        self.conv.groups)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            ret = self.forward_concrete(x)
        else:
            ret = self.forward_abstract(x)
        return ret


class Sequential(nn.Module):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward_until(self, i, x):
        for layer in self.layers[:i + 1]:
            x = layer(x)
        return x

    def forward_from(self, i, x):
        for layer in self.layers[i + 1:]:
            x = layer(x)
        return x

    def total_abs_l1(self, x):
        ret = 0
        for layer in self.layers:
            x = layer(x)
            ret += x.l1()
        return ret

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(self, x, init_lambda=False, skip_norm=False):
        for layer in self.layers:
            if isinstance(layer, Normalization) and skip_norm:
                continue
            if isinstance(layer, ReLU):
                x = layer(x, init_lambda)
            else:
                x = layer(x)
        return x


class ReLU(nn.Module):

    def __init__(self, dims=None):
        super(ReLU, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(torch.ones(dims))
        self.bounds = None

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x, init_lambda=False):
        return x.relu()


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view((x.size()[0], -1))


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.linear(x)
        else:
            return x.linear(self.linear.weight, self.linear.bias)


class Normalization(nn.Module):

    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.sigma
        ret = x.normalize(self.mean, self.sigma)
        return ret


# class Classifier(nn.Module):
#     def __init__(self, model=None):
#         super(Classifier, self).__init__()
#         if model is not None:
#             self.model = model
#         else:
#             self.model = None
#
#     def forward(self, x):
#         if self.model is None:
#             raise NotImplementedError
#         else:
#             return self.model.forward(x)
#
#     def predict(self, x):
#         return [self.forward(x)]
#
#     def fit(self, loader, loss_fcn, optimizer, nb_epochs=10, **kwargs):
#
#         if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
#             reduce_labels = True
#         else:
#             assert 0
#         # Start training
#         for i in range(nb_epochs):
#             pbar = tqdm(loader)
#             for i_batch, o_batch in pbar:
#                 i_batch, o_batch = i_batch.to('cuda'), o_batch.to('cuda')
#                 optimizer.zero_grad()
#                 # Perform prediction
#                 model_outputs = self.forward(i_batch)
#                 # Form the loss function
#                 loss = loss_fcn(model_outputs, o_batch)
#                 loss.backward()
#                 optimizer.step()
#                 pbar.set_description("epoch {:d}".format(i))
#
#     def advfit(self, loader, loss_fcn, optimizer, attack, epsilon, nb_epochs=10, ratio=0.5, **kwargs):
#         import foolbox as fb
#
#         assert (0 <= ratio <= 1), "ratio must be between 0 and 1"
#         if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
#             reduce_labels = True
#         else:
#             assert 0
#
#         # Start training
#         for _ in range(nb_epochs):
#             pbar = tqdm(loader)
#             # Shuffle the examples
#             for i_batch, o_batch in pbar:
#                 i_batch, o_batch = i_batch.to('cuda'), o_batch.to('cuda')
#
#                 self.eval()
#                 fmodel = fb.PyTorchModel(self, bounds=(0, 1))
#                 adv_batch, _ = attack(fmodel, i_batch, o_batch, epsilon=epsilon, **kwargs)
#                 self.train()
#
#                 optimizer.zero_grad()
#                 # Perform prediction
#                 model_outputs = self.forward(i_batch)
#                 adv_outputs = self.forward(adv_batch)
#                 loss = (1 - ratio) * loss_fcn(model_outputs, o_batch) + ratio * loss_fcn(adv_outputs, o_batch)
#
#                 # Actual training
#                 loss.backward()
#                 optimizer.step()
#             # pbar.set_description()
#
#     def save(self, filename, path):
#         """
#         Save a model to file in the format specific to the backend framework.
#         :param filename: Name of the file where to store the model.
#         :type filename: `str`
#         :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
#                      the default data location of the library `ART_DATA_PATH`.
#         :type path: `str`
#         :return: None
#         """
#         import os
#         assert (path is not None)
#
#         full_path = os.path.join(path, filename)
#         folder = os.path.split(full_path)[0]
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#         torch.save(self.state_dict(), full_path + ".model")
#
#     def load(self, path):
#         self.load_state_dict(torch.load(path))
#
#     def freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def unfreeze(self):
#         for param in self.parameters():
#             param.requires_grad = False


def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    elif dataset == 'imagenet32' or dataset == 'imagenet64':
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
    else:
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
    return mean.to(device), sigma.to(device)


class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = False


class FFNN(SeqNet):

    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3):
        super(FFNN, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [Flatten(), Linear(input_size * input_size * input_channel, sizes[0]), ReLU(sizes[0])]
        for i in range(1, len(sizes)):
            layers += [
                Linear(sizes[i - 1], sizes[i]),
                ReLU(sizes[i]),
            ]
        layers += [Linear(sizes[-1], n_class)]
        self.blocks = Sequential(*layers)


class ConvMed(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1,
                 linear_size=100):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16 * width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16 * width1, input_size // 2, input_size // 2)),
            Conv2d(16 * width1, 32 * width2, 4, stride=2, padding=1, dim=input_size // 2),
            ReLU((32 * width2, input_size // 4, input_size // 4)),
            Flatten(),
            Linear(32 * width2 * (input_size // 4) * (input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


class ConvMedBatchNorm(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1,
                 linear_size=100):
        super(ConvMedBatchNorm, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16 * width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16 * width1, input_size // 2, input_size // 2)),
            nn.BatchNorm2d(16 * width1),
            Conv2d(16 * width1, 32 * width2, 4, stride=2, padding=1, dim=input_size // 2),
            ReLU((32 * width2, input_size // 4, input_size // 4)),
            nn.BatchNorm2d(32 * width2),
            Flatten(),
            Linear(32 * width2 * (input_size // 4) * (input_size // 4), linear_size),
            ReLU(linear_size),
            nn.BatchNorm1d(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=4, width2=4, width3=2,
                 linear_size=200, with_normalization=True):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        if with_normalization:
            layers = [
                Normalization(mean, sigma),
                Conv2d(input_channel, 16 * width1, 3, stride=1, padding=1, dim=input_size),
                ReLU((16 * width1, input_size, input_size)),
                Conv2d(16 * width1, 16 * width2, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((16 * width2, input_size // 2, input_size // 2)),
                Conv2d(16 * width2, 32 * width3, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((32 * width3, input_size // 4, input_size // 4)),
                Flatten(),
                Linear(32 * width3 * (input_size // 4) * (input_size // 4), linear_size),
                ReLU(linear_size),
                Linear(linear_size, n_class),
            ]
        else:
            layers = [
                Conv2d(input_channel, 16 * width1, 3, stride=1, padding=1, dim=input_size),
                ReLU((16 * width1, input_size, input_size)),
                Conv2d(16 * width1, 16 * width2, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((16 * width2, input_size // 2, input_size // 2)),
                Conv2d(16 * width2, 32 * width3, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((32 * width3, input_size // 4, input_size // 4)),
                Flatten(),
                Linear(32 * width3 * (input_size // 4) * (input_size // 4), linear_size),
                ReLU(linear_size),
                Linear(linear_size, n_class),
            ]
        self.blocks = Sequential(*layers)
        self.layers = layers


class ConvMedBig1(SeqNet):
    def __init__(self, convmedbig):
        super(ConvMedBig1, self).__init__()
        assert (isinstance(convmedbig, ConvMedBig)), "This wrapper takes convmedbig model only"
        self.blocks = Sequential(*convmedbig.layers[:-2])


class ConvMedBig2(SeqNet):
    def __init__(self, convmedbig):
        super(ConvMedBig2, self).__init__()
        assert (isinstance(convmedbig, ConvMedBig)), "This wrapper takes convmedbig model only"
        self.blocks = Sequential(*convmedbig.layers[-2:])


# class SimpleNet(Classifier):
#     def __init__(self, in_ch, out_ch):
#         super(SimpleNet, self).__init__()
#         self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=4, kernel_size=5, stride=1)
#         self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
#         self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
#         self.fc_2 = nn.Linear(in_features=100, out_features=out_ch)
#         self.nclasses = out_ch
#
#     def forward(self, x):
#         x = F.relu(self.conv_1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 10)
#         x = F.relu(self.fc_1(x))
#         x = self.fc_2(x)
#         return x
#
#     def forwardToDetect(self, x):
#         x = F.relu(self.conv_1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 10)
#         x = F.relu(self.fc_1(x))
#         return x
#
#
# class SimpleNet1(Classifier):
#     def __init__(self, in_ch, out_ch):
#         super(SimpleNet1, self).__init__()
#         self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=4, kernel_size=5, stride=1)
#         self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
#         self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
#
#     # self.fc_2 = nn.Linear(in_features=100, out_features=out_ch)
#     # self.nclasses = out_ch
#
#     def forward(self, x):
#         x = F.relu(self.conv_1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 10)
#         x = F.relu(self.fc_1(x))
#         return x
#
#
# class SimpleNet2(Classifier):
#     def __init__(self, in_ch, out_ch):
#         super(SimpleNet2, self).__init__()
#         self.fc_2 = nn.Linear(in_features=100, out_features=out_ch)
#         self.nclasses = out_ch
#
#     def forward(self, x):
#         x = self.fc_2(x)
#         return x
#
#
# class SimpleDetector(Classifier):
#     def __init__(self, dmodel, in_features=100):
#         super(SimpleDetector, self).__init__()
#         self.fc1 = nn.Linear(in_features=in_features, out_features=2)
#         self.param = [self.fc1.weight, self.fc1.bias]
#
#     def forward(self, x):
#         x = self.fc1(x)
#         return x
#
#
# class SimpleEnsemble(Classifier):
#     def __init__(self, in_ch, out_ch, N):
#         super(SimpleEnsemble, self).__init__()
#         self.N = N
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.modelList = []
#         for i in range(N):
#             self.modelList.append(Classifier(SimpleNet(in_ch, out_ch)))
#
#     def forward(self, x):
#         pred = torch.zeros(x.shape[0], self.out_ch)
#         for model in self.modelList:
#             pred += model(x)
#         return pred
