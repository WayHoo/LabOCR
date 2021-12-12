# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):
    pass


def read_params():
    cfg = Config()

    # params for text detector
    cfg.det_algorithm = "DB"
    cfg.det_model_dir = "./inference/ch_det_infer/"
    cfg.det_limit_side_len = 960
    cfg.det_limit_type = 'max'

    # DB parmas
    cfg.det_db_thresh = 0.3
    cfg.det_db_box_thresh = 0.5
    cfg.det_db_unclip_ratio = 1.6
    cfg.use_dilation = False
    cfg.det_db_score_mode = "fast"

    # params for text recognizer
    cfg.rec_algorithm = "CRNN"
    cfg.rec_model_dir = "./inference/ch_rec_infer/"

    cfg.rec_image_shape = "3, 32, 320"
    cfg.rec_char_type = 'ch'
    cfg.rec_batch_num = 30
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "./ppocr/utils/ocr_keys.txt"
    cfg.use_space_char = True

    # params for text classifier
    cfg.use_angle_cls = True
    cfg.cls_model_dir = "./inference/ch_cls_infer/"
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    # params for angle detector
    cfg.use_angle_det = True
    cfg.agl_det_model_dir = "./inference/angle_det/"
    cfg.use_size_adjust = False
    cfg.adjust_thresh = 0.05

    # params for table recognizer
    cfg.sub_table_thresh = 0.5  # 化验单栏数判断阈值
    cfg.head_word_seg_thresh = 0.75  # 化验单表头文本框分词阈值
    cfg.horizon_line_split_thresh = 0.6  # 行切割阈值

    cfg.use_pdserving = False
    cfg.use_tensorrt = False
    cfg.drop_score = 0.5

    return cfg
