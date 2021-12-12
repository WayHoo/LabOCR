import copy
import paddle
import paddle.nn as nn

# det loss
from .det_db_loss import DBLoss

# rec loss
from .rec_ctc_loss import CTCLoss
# cls loss
from .cls_loss import ClsLoss

# basic loss function
from .basic_loss import DistanceLoss

# combined loss function
from .combined_loss import CombinedLoss

def build_loss(config):
    support_dict = [
        'DBLoss', 'CTCLoss', 'ClsLoss', 'CombinedLoss'
    ]

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
