from functools import reduce

import tensorflow as tf
from secml.array import CArray
from secml.ml import CClassifierDNN


class CClassifierKeras(CClassifierDNN):
    """CClassifierKeras, wrapper for Keras models.

        Parameters
        ----------
        model : model dtype of the specific backend
            The model to wrap.
        input_shape : tuple or None, optional
            Shape of the input for the DNN, it will
            be used for reshaping the input data to
            the expected shape.
        preprocess : CPreprocess or str or None, optional
            Preprocessing module.
        pretrained : bool, optional
            Whether or not the model is pretrained. If the
            model is pretrained, the user won't need to call
            `fit` after loading the model. Default False.
        pretrained_classes : None or CArray, optional
            List of classes labels if the model is pretrained. If
            set to None, the class labels for the pretrained model should
            be inferred at the moment of initialization of the model
            and set to CArray.arange(n_classes). Default None.
        softmax_outputs : bool, optional
            Whether or not to add a softmax layer after the
            logits. Default False.
        n_jobs : int, optional
            Number of parallel workers to use for training the classifier.
            Cannot be higher than processor's number of cores. Default is 1.

        Attributes
        ----------
        class_type : 'keras-clf'

        """
    __class_type = 'keras-clf'

    def __init__(self, model, input_shape=None, preprocess=None,
                 pretrained=False, pretrained_classes=None,
                 softmax_outputs=False, n_jobs=1):
        super(CClassifierKeras, self).__init__(
            model=model, input_shape=input_shape, preprocess=preprocess,
            pretrained=pretrained, pretrained_classes=pretrained_classes,
            softmax_outputs=softmax_outputs, n_jobs=n_jobs)

        self._cached_in = None
        self._cached_out = None
        self._tape = tf.GradientTape(persistent=True)

        if self._pretrained is True:
            self._trained = True
            if self._pretrained_classes is not None:
                self._classes = self._pretrained_classes
            else:
                self._classes = CArray.arange(
                    reduce(lambda x, y: x * y, self._model.output_shape[1:]))
            self._n_features = reduce(lambda x, y: x * y,
                                      self._model.input_shape[1:])

    @property
    def layers(self):
        """Returns list of tuples containing the layers of the model.
        Each tuple is structured as (layer_name, layer)."""

        # excluding input layer
        return [(layer.name, layer) for layer in self._model.layers[1:]]

    @property
    def layer_shapes(self):
        """Returns a dictionary containing the shapes of the output
        of each layer of the model."""

        # excluding input layer
        return {layer.name: layer.output_shape[1:]
                for layer in self._model.layers[1:]}

    @staticmethod
    def _to_tensor(x, shape=None):
        """Convert input CArray to backend-supported tensor."""
        tensor = tf.convert_to_tensor(x.tondarray(), dtype=tf.float32)
        if shape is not None:
            tensor = tf.reshape(tensor, shape)
        return tensor

    @staticmethod
    def _from_tensor(x):
        """Convert input backend-supported tensor to CArray"""
        return CArray(x.numpy())

    def _forward(self, x):
        """Forward pass on input x.
        Returns the output of the layer set in _out_layer.
        If _out_layer is None, the last layer output is returned,
        after applying softmax if softmax_outputs is True.

        Parameters
        ----------
        x : CArray
            preprocessed array, ready to be transformed by the current module.

        Returns
        -------
        CArray
            Transformed input data.
        """
        x = self._to_tensor(x, shape=(-1, *self._input_shape))

        if self._cached_x is None:
            # grad is not required
            if self._tape._recording:
                self._tape._pop_tape()
                self._tape._tape = None
            self._cached_in = None
            self._cached_out = None
            out = self._model(x)
        else:
            if self._tape._recording:
                self._tape.reset()
            else:
                self._tape.__enter__()
            self._tape.watch(x)
            out = self._model(x)
            self._cached_in = x
            self._cached_out = [out[:, i] for i in self.classes]
            self._tape.stop_recording()

        return self._from_tensor(out)

    def _backward(self, w):
        """Returns the gradient of the DNN - considering the output layer set
        in _out_layer - wrt data.

        Parameters
        ----------
        w : CArray
            Weights that are pre-multiplied to the gradient
            of the module, as in standard reverse-mode autodiff.

        Returns
        -------
        gradient : CArray
            Accumulated gradient of the module wrt input data.
        """
        if w is None:
            w = CArray.ones(shape=(self.n_classes,))

        if not self._cached_x.is_vector_like:
            raise ValueError("Tested only on vector-like arrays")
        grad = CArray.zeros(shape=self._cached_x.shape)

        # loop only over non-zero elements in w, to save computations
        for c in w.nnz_indices[1]:
            grad += w[c] * self._from_tensor(
                self._tape.gradient(self._cached_out[c], self._cached_in))
        self._tape._pop_tape()
        self._tape._tape = None
        return grad

    def save_model(self, filename):
        """
        Stores the model and optimization parameters.

        Parameters
        ----------
        filename : str
            path of the file for storing the model

        """
        pass

    def load_model(self, filename):
        """
        Restores the model and optimization parameters.
        Notes: the model class should be
        defined before loading the params.

        Parameters
        ----------
        filename : str
            path where to find the stored model

        """
        pass

    def _fit(self, x, y):
        """Private method that trains the One-Vs-All classifier.
        Must be reimplemented by subclasses.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray or None, optional
            Array of shape (n_samples,) containing the class labels.
            Can be None if not required by the algorithm.

        Returns
        -------
        CClassifier
            Trained classifier.

        """
        pass
