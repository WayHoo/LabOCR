__all__ = ['build_head']


def build_head(config):
    # det head
    from .det_db_head import DBHead

    # rec head
    from .rec_ctc_head import CTCHead

    # cls head
    from .cls_head import ClsHead
    support_dict = [
        'DBHead', 'CTCHead', 'ClsHead'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
