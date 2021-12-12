import os
import sys
import json
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import copy
import numpy as np
import time
import logging
import traceback
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
import tools.infer.predict_agl as predict_agl
from kie.tsr.table_rec import TableRecognizer
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.use_angle_det = args.use_angle_det
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)
        if self.use_angle_det:
            self.angle_detector = predict_agl.AngleDetector(args)

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img, cls=True):
        if self.use_angle_det:
            angle, img = self.angle_detector(img)

        round_idx, max_round = 0, 2
        has_rotated = False
        while round_idx < max_round:
            round_idx += 1
            ori_im = img.copy()
            dt_boxes, elapse = self.text_detector(img)

            logger.debug("dt_boxes num: {}, elapse: {}".format(len(dt_boxes), elapse))
            if dt_boxes is None:
                return None, None
            img_crop_list, det_rotate_list = [], []
            dt_boxes = sorted_boxes(dt_boxes)

            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop, rotated = get_rotate_crop_image(ori_im, tmp_box)
                img_crop_list.append(img_crop)
                det_rotate_list.append(rotated)
            if self.use_angle_cls and cls:
                img_crop_list, angle_list, cls_rotate, elapse = \
                    self.text_classifier(img_crop_list, det_rotate_list)
                logger.debug("cls num: {}, cls_rotate: {}°, elapse: {}".
                    format(len(img_crop_list), cls_rotate, elapse))
                
                if self.use_angle_det and cls_rotate != 0 and not has_rotated:
                    rotate_param = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}
                    img = cv2.rotate(img, rotate_param[cls_rotate])
                    has_rotated = True
                    continue

            rec_res, elapse = self.text_recognizer(img_crop_list)
            logger.debug("rec_res num: {}, elapse: {}".format(len(rec_res), elapse))
            # self.print_draw_crop_rec_res(img_crop_list, rec_res)
            filter_boxes, filter_rec_res = [], []
            for box, rec_reuslt in zip(dt_boxes, rec_res):
                text, score = rec_reuslt
                if score >= self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_reuslt)
            return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    # num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    # for i in range(num_boxes - 1):
    #     if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
    #             (_boxes[i + 1][0][0] < _boxes[i][0][0]):
    #         tmp = _boxes[i]
    #         _boxes[i] = _boxes[i + 1]
    #         _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    table_rec = TableRecognizer(args)
    is_visualize = True
    write_raw_items = True  # 是否将未纠错的项目值写入文件
    font_path = args.vis_font_path
    drop_score = args.drop_score

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    begin = time.time()
    error_img_list = []
    total_boxes = 0  # 检测文本框数量
    raw_items = []  # 未纠错的项目值
    for idx, image_file in enumerate(image_file_list):
        try:
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            start_time = time.time()
            dt_boxes, rec_res = text_sys(img)
            elapse = time.time() - start_time
            total_boxes += len(dt_boxes)

            logger.info(str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse))
            for text, score in rec_res:
                logger.info("{}, {:.3f}".format(text, score))

            # table structure recognition and ocr correct
            img_name = os.path.basename(image_file).split(".")[0]
            _, entries, items = table_rec(img, img_name, dt_boxes, rec_res, save=True)
            print("------------------entries begin------------------")
            print(json.dumps(entries, ensure_ascii=False))
            print("------------------entries end------------------")
            raw_items.append(os.path.basename(image_file))
            raw_items.extend(items)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(image, boxes, txts, scores, 
                    drop_score=drop_score, font_path=font_path)
                draw_img_save = args.save_path
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                if flag:
                    image_file = image_file[:-3] + "png"
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename(image_file)),
                    draw_img[:, :, ::-1])
                logger.info("The visualized image saved in {}".format(
                    os.path.join(draw_img_save, os.path.basename(image_file))))
        except BaseException as e:
            error_img_list.append(image_file)
            logger.error("Exception occurred: {}".format(e))
            logger.error(traceback.format_exc())
            continue
    # 将未纠错的项目值写入文件
    if write_raw_items:
        item_save_path = os.path.join(args.save_path, "items.txt")
        with open(item_save_path, mode='w') as fw:
            for item in raw_items:
                fw.write(item+"\n")
    print('----------------image process statistic----------------')
    elapesd = time.time() - begin
    print('total image num:', len(image_file_list))
    print('error image num:', len(error_img_list))
    print('error image list:', error_img_list)
    print('predict total time: %.2fs' % elapesd)
    print('predict avg time: %.2fs' % (elapesd/len(image_file_list)))
    print('total boxes:', total_boxes)


if __name__ == "__main__":
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
