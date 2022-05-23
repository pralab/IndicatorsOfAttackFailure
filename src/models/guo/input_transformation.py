from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import kornia
import torch.nn

logger = logging.getLogger(__name__)


class InputTransformation(torch.nn.Module):
	params = ["degrees", "translate", "scales"]

	def __init__(self, degrees=(-40, 40), translate=None, scales=None):
		super(InputTransformation, self).__init__()
		self.degrees = degrees
		self.translate = translate
		self.scales = scales

	def forward(self, x):
		my_fcn = kornia.augmentation.RandomAffine(self.degrees, self.translate, self.scales, return_transform=False)
		return my_fcn(x)
