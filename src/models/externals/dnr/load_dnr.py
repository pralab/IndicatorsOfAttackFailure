import os

import torchvision
from robustbench.utils import download_gdrive
from secml.array import CArray
from secml.ml import CClassifierSVM, CKernelRBF, CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierDNR

MODEL_ID = "1vMYt0SG8-WKQM-DuELtZCzqQjfdKx-2V"


def load_model():
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1e-4)
    layer_23 = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=10)
    layer_26 = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=1)
    layer_29 = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=0.1)

    dnn = torchvision.models.vgg19(pretrained=True)
    dnn.eval()
    dnn = CClassifierPyTorch(model=dnn, input_shape=(3, 32, 32), pretrained=True, pretrained_classes=CArray(range(10)))

    layers_clf = {'features:23': layer_23, 'features:26': layer_26, 'features:29': layer_29}
    model = CClassifierDNR(combiner=combiner, layer_clf=layers_clf, dnn=dnn,
                           layers=list(layers_clf.keys()), threshold=-1000)
    path = os.path.join(os.path.dirname(__file__), 'dnr_cifar_net_rejection.gz')
    if not os.path.exists(path):
        download_gdrive(MODEL_ID, path)
    model = model.load(path)
    return model
