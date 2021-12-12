from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_dygraph_params
import tools.program as program


def main():
    global_config = config['Global']
    # build dataloader
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                config['Architecture']["Models"][key]["Head"][
                    'out_channels'] = char_num
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

    model = build_model(config['Architecture'])

    best_model_dict = load_dygraph_params(config, model, logger, None)
    if len(best_model_dict):
        logger.info('metric in ckpt ***************')
        for k, v in best_model_dict.items():
            logger.info('{}:{}'.format(k, v))

    # build metric
    eval_class = build_metric(config['Metric'])

    # start eval
    metric = program.eval(model, valid_dataloader, post_process_class, eval_class)
    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
