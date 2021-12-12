from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle


class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.regularization_coeff = factor

    def __call__(self):
        reg = paddle.regularizer.L1Decay(self.regularization_coeff)
        return reg


class L2Decay(object):
    """
    L2 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.regularization_coeff = factor

    def __call__(self):
        reg = paddle.regularizer.L2Decay(self.regularization_coeff)
        return reg
