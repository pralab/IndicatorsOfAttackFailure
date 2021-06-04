import os

import torch
from robustbench.utils import download_gdrive
from secml.array import CArray
from secml.ml import CClassifierPyTorch

from src.models.kwta.models import SparseResNet18

MODEL_ID = '1Af_owmMvg1LxjITLE1gFUmPx5idogeTP'


def load_model():
    gamma = 0.1
    filepath = os.path.join(os.path.dirname(__file__), f'kwta_spresnet18_{gamma}_cifar_adv.pth')
    if not os.path.exists(filepath):
        download_gdrive(MODEL_ID, filepath)
    model = SparseResNet18(sparsities=[gamma, gamma, gamma, gamma], sparse_func='vol')
    if not torch.cuda.is_available():
        state_dict = torch.load(filepath, map_location='cpu')
    else:
        state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    model.eval()
    clf = CClassifierPyTorch(model, input_shape=(3, 32, 32), pretrained=True,
                       pretrained_classes=CArray(list(range(10))), preprocess=None)
    return clf
