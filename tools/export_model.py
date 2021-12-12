import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

import argparse

import paddle
from paddle.jit import to_static

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_dygraph_params
from ppocr.utils.logging import get_logger
from tools.program import load_config, merge_config, ArgsParser


def export_single_model(model, arch_config, save_path, logger):
    infer_shape = [3, -1, -1]
    if arch_config["model_type"] == "rec":
        infer_shape = [3, 32, -1]  # for rec model, H must be 32
    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype="float32")
        ])

    paddle.jit.save(model, save_path)
    logger.info("inference model is saved to {}".format(save_path))
    return


def main():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    logger = get_logger()
    # build post process

    post_process_class = build_post_process(config["PostProcess"],
                                            config["Global"])

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                config["Architecture"]["Models"][key]["Head"][
                    "out_channels"] = char_num
                # just one final tensor needs to to exported for inference
                config["Architecture"]["Models"][key][
                    "return_all_feats"] = False
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num
    model = build_model(config["Architecture"])
    _ = load_dygraph_params(config, model, logger, None)
    model.eval()

    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if arch_config["algorithm"] in ["Distillation", ]:  # distillation model
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(model.model_list[idx], archs[idx],
                                sub_model_save_path, logger)
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(model, arch_config, save_path, logger)


if __name__ == "__main__":
    main()
