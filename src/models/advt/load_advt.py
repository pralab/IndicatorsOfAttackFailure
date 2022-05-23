import robustbench
from secml.array import CArray
from secml.ml import CClassifierPyTorch


def load_model():
    model = robustbench.load_model('Engstrom2018Robustness')
    clf = CClassifierPyTorch(model, input_shape=(3, 32, 32), pretrained=True,
                             pretrained_classes=CArray(list(range(10))), preprocess=None)
    return clf
