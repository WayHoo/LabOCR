from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class CTCLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def forward(self, predicts, batch):
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        loss = loss.mean()  # sum
        return {'loss': loss}
