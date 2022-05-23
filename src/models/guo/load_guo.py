import os.path

import torch.nn
from robustbench.utils import download_gdrive
from secml.array import CArray
from secml.ml import CClassifierPyTorch

from src.models.common.dag_module import DAGModule
from src.models.common.net import ConvMedBig
from src.models.guo.input_transformation import InputTransformation

MODEL_ID = "107e6eY5buz8wqo-N_zFrQkAFd1X6V38I"


def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    it_defense = InputTransformation(degrees=(-30, 30))
    network1 = ConvMedBig(device=device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200,
                          input_channel=3, with_normalization=True)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    if not os.path.exists(model_path):
        download_gdrive(MODEL_ID, model_path)
    state_dict = torch.load(model_path, map_location=device)
    classifier = DAGModule([it_defense, network1], device=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    clf = CClassifierPyTorch(classifier, input_shape=(3, 32, 32), pretrained=True,
                             pretrained_classes=CArray(list(range(10))), preprocess=None)
    return clf
