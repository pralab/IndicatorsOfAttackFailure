import math
from functools import reduce

import torch
from secml.core.exceptions import NotFittedError
from secml.ml import CClassifierPyTorch


class CClassifierPytorchYuHu(CClassifierPyTorch):
    def __init__(self, model, input_shape=None, preprocess=None,
                 batch_size=1, n_jobs=1, threshold=None, hide_reject=False):
        super(CClassifierPytorchYuHu, self).__init__(
            model=model, loss=None, optimizer=None, optimizer_scheduler=None,
            pretrained=True, pretrained_classes=None, input_shape=input_shape,
            random_state=None, preprocess=preprocess, softmax_outputs=False,
            epochs=10, batch_size=batch_size, n_jobs=n_jobs)

        self._noise_radius = 1
        self._threshold = threshold
        self._hide_reject = hide_reject
        self._cached_detection_scores = None

    @property
    def threshold(self):
        """Returns the rejection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """Sets the rejection threshold."""
        self._threshold = float(value)

    @property
    def hide_reject(self):
        return self._hide_reject

    @hide_reject.setter
    def hide_reject(self, value):
        self._hide_reject = value

    @property
    def classes(self):
        """Return the list of classes on which training has been performed."""
        if self._hide_reject:
            return super(CClassifierPytorchYuHu, self).classes
        else:
            return super(CClassifierPytorchYuHu, self).classes.append([-1])

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
        data_loader = self._data_loader(x, num_workers=self.n_jobs - 1,
                                        batch_size=self._batch_size)

        # Switch to evaluation mode
        self._model.eval()
        n_classes = self.n_classes if self._hide_reject else self.n_classes - 1
        out_shape = n_classes if self._out_layer is None else \
            reduce((lambda z, v: z * v), self.layer_shapes[self._out_layer])
        output = torch.empty((len(data_loader.dataset), out_shape))
        output_n = torch.empty((len(data_loader.dataset), out_shape))

        for batch_idx, (s, _) in enumerate(data_loader):
            # Log progress
            self.logger.info(
                'Classification: {batch}/{size}'.format(batch=batch_idx,
                                                        size=len(data_loader)))

            s = s.to(self._device)
            s_n = self._add_noise(s)

            if self._cached_x is None:
                self._cached_s = None
                self._cached_layer_output = None
                self._cached_detection_scores = None
                with torch.no_grad():
                    ps = self._get_layer_output(s, self._out_layer)
                    ps_n = self._get_layer_output(s_n, self._out_layer)

            else:
                # keep track of the gradient in s tensor
                s.requires_grad = True
                ps = self._get_layer_output(s, self._out_layer)
                ps_n = self._get_layer_output(s_n, self._out_layer)
                self._cached_s = s
                self._cached_layer_output = ps

            output[batch_idx * self.batch_size:
                   batch_idx * self.batch_size + len(s)] = \
                ps.view(ps.size(0), -1)
            output_n[batch_idx * self.batch_size:
                     batch_idx * self.batch_size + len(s_n)] = \
                ps_n.view(ps_n.size(0), -1)

        scores = output.detach()
        scores = self._from_tensor(scores)

        if not self._hide_reject:
            detection_scores = torch.norm(
                output.softmax(dim=1) - output_n.softmax(dim=1), p=1, dim=1)
            if self._cached_x is not None:
                self._cached_detection_scores = detection_scores
            detection_scores = self._from_tensor(
                detection_scores.detach()).ravel()
            rej_idx = detection_scores > self._threshold
            detection_scores[rej_idx.logical_not()] = \
                scores[rej_idx.logical_not(), :].min(axis=1).ravel() - \
                abs(detection_scores[rej_idx.logical_not()])
            detection_scores[rej_idx] = \
                scores[rej_idx, :].max(axis=1).ravel() + \
                abs(detection_scores[rej_idx])
            scores = scores.append(detection_scores.T, axis=1)

        return scores

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
            raise ValueError("Function `_backward` needs the `w` array "
                             "to run backward with.")
        if not w.is_vector_like or not self._cached_x.is_vector_like:
            raise ValueError("Gradient can be computed for only one sample")

        if w.atleast_2d().shape[1] != self.n_classes:
            raise ValueError("The shape of w must be equal to "
                             "classifier output")

        if self._hide_reject:
            w_out = self._to_tensor(w.atleast_2d()).reshape(
                self._cached_layer_output.shape)
        else:
            w_out = self._to_tensor(w[:-1].atleast_2d()).reshape(
                self._cached_layer_output.shape)
        w_out = w_out.to(self._device)

        if self._cached_s.grad is not None:
            self._cached_s.grad.zero_()

        self._cached_layer_output.backward(w_out, retain_graph=True)

        grad = self._from_tensor(self._cached_s.grad.data.view(
            -1, reduce(lambda a, b: a * b, self.input_shape)))

        if w[-1] != 0 and not self._hide_reject:
            w_d = self._to_tensor(w[-1].atleast_2d()).reshape(
                self._cached_detection_scores.shape)
            w_d = w_d.to(self._device)

            if self._cached_s.grad is not None:
                self._cached_s.grad.zero_()

            self._cached_detection_scores.backward(w_d)

            grad_d = self._from_tensor(self._cached_s.grad.data.view(
                -1, reduce(lambda a, b: a * b, self.input_shape)))
            if grad_d.norm() > 1e-20:
                grad_d /= grad_d.norm()
                grad_d *= 100.
            grad += grad_d

        return grad

    def _add_noise(self, s):
        return s + self._noise_radius * torch.rand_like(s)

    def predict(self, x, return_decision_function=False):
        """Perform classification of each pattern in x.

        If preprocess has been specified,
        input is normalized before classification.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the `decision_function` value along
            with predictions. Default False.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
            to each test pattern. The classification label is the label of
            the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
            score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.
        """
        labels, scores = super(CClassifierPytorchYuHu, self).predict(
            x, return_decision_function=True)

        if not self._hide_reject:
            labels[labels == self.n_classes - 1] = -1
        return (labels, scores) if return_decision_function is True else labels

    def compute_threshold(self, rej_percent, ds):
        """Compute the threshold that must be set in the classifier to have
        rej_percent rejection rate (accordingly to an estimation on a
        validation set).

        Parameters
        ----------
        rej_percent : float
            Max percentage of rejected samples.
        ds : CDataset
            Dataset on which the threshold is estimated.

        Returns
        -------
        threshold : float
            The estimated reject threshold

        """
        if not self.is_fitted():
            raise NotFittedError("The classifier must be fitted")

        data_loader = self._data_loader(ds.X, num_workers=self.n_jobs - 1,
                                        batch_size=self._batch_size)

        # Switch to evaluation mode
        self._model.eval()
        n_classes = self.n_classes if self._hide_reject else self.n_classes - 1
        out_shape = n_classes if self._out_layer is None else \
            reduce((lambda z, v: z * v), self.layer_shapes[self._out_layer])
        output = torch.empty((len(data_loader.dataset), out_shape))
        output_n = torch.empty((len(data_loader.dataset), out_shape))

        for batch_idx, (s, _) in enumerate(data_loader):
            s = s.to(self._device)
            s_n = self._add_noise(s)
            with torch.no_grad():
                ps = self._get_layer_output(s, self._out_layer)
                ps_n = self._get_layer_output(s_n, self._out_layer)
            output[batch_idx * self.batch_size:
                   batch_idx * self.batch_size + len(s)] = \
                ps.view(ps.size(0), -1)
            output_n[batch_idx * self.batch_size:
                     batch_idx * self.batch_size + len(s_n)] = \
                ps_n.view(ps_n.size(0), -1)

        scores = output.detach()

        detection_scores = torch.norm(
            output.softmax(dim=1) - output_n.softmax(dim=1), p=1, dim=1)
        if self._cached_x is not None:
            self._cached_detection_scores = detection_scores
        detection_scores = self._from_tensor(
            detection_scores.detach()).ravel().sort()[::-1]
        rej_num = math.floor(rej_percent * ds.num_samples)
        threshold = detection_scores[rej_num - 1].item()
        self.logger.info("Chosen threshold: {:}".format(threshold))
        return threshold

    def _check_clf_index(self, y):
        """Raise error if index y is outside [-1, n_classes) range.

        Parameters
        ----------
        y : int
            class label index.

        """
        if y < -1 or y >= self.n_classes:
            raise ValueError(
                "class label {:} is out of range".format(y))
