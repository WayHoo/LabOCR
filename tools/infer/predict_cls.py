import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import copy
import numpy as np
import math
import time
import traceback

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif

logger = get_logger()


class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, _ = \
            utility.create_predictor(args, 'cls', logger)

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def check_img_rotate(self, det_rotate_list, cls_rotate_list):
        """
        check if the image need rotate and go back to run text detector again
        :param det_rotate_list: True - rotate 90° counterclockwise when aspect ratio is greater than 1.5
        :param cls_rotate_list: True - rotate 180° in cls_det
        :return: the angle that needs to be rotated, [0°, 90°, 180°, 270°]
        """
        degree = [0, 90, 180, 270]
        rotate_list = [0] * 4
        for i, det_ro in enumerate(det_rotate_list):
            cls_ro = cls_rotate_list[i]
            if not det_ro and not cls_ro:
                rotate_list[0] += 1  # no need to rotate
            elif not det_ro and cls_ro:
                rotate_list[2] += 1  # rotate 180°
            elif det_ro and not cls_ro:
                rotate_list[3] += 1  # rotate 90° counterclockwise
            else:
                rotate_list[1] += 1  # # rotate 270° counterclockwise
        max_idx = rotate_list.index(max(rotate_list))
        return degree[max_idx]

    def __call__(self, img_list, det_rotate_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        cls_rotate_list = [False] * img_num
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            start_time = time.time()

            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()
            prob_out = self.output_tensors[0].copy_to_cpu()
            self.predictor.try_shrink_memory()
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - start_time
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                cls_rotate = False
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], cv2.ROTATE_180)
                    cls_rotate = True
                cls_rotate_list[indices[beg_img_no + rno]] = cls_rotate
        # check if the image need rotate and go back to run text detector again
        final_rotate = self.check_img_rotate(det_rotate_list, cls_rotate_list)
        return img_list, cls_res, final_rotate, elapse


def main(args):
    args.image_dir = "./train_data/TestSheetRecEval/crop_img/"
    args.use_gpu = True
    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = TextClassifier(args)
    valid_image_file_list = []
    img_list = []
    aspect_ratio_list = []
    begin = time.time()
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
        # 计算图像长宽比
        size = img.shape
        w, h = size[1], size[0]
        aspect_ratio_list.append(w/h)
    try:
        det_rotate_list = [False] * len(img_list)
        img_list, cls_res, final_rotate, predict_time = \
            text_classifier(img_list, det_rotate_list)
    except:
        logger.info(traceback.format_exc())
        logger.info(
            "ERROR!!!! \n"
            "Please read the FAQ：https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' ")
        exit()
    end = time.time()
    err_cnt = 0
    err_aspect_ratio_list = []
    for ino in range(len(img_list)):
        if cls_res[ino][0] != '0':
            err_cnt += 1
            err_aspect_ratio_list.append(aspect_ratio_list[ino])
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))
    print("-------------------predict_cls statistic-------------------")
    total = len(img_list)
    accuracy = 100*(total-err_cnt)/total
    elapsed = (end - begin) * 1000
    err_aspect_ratio_list = sorted(err_aspect_ratio_list)
    print("total img: %d, error img: %d, accuracy: %.2f" % (total, err_cnt, accuracy))
    print("time elapsed: %.2fms, avg time cost: %.2fms" % (elapsed, elapsed/total))
    print("err_aspect_ratio_list:", err_aspect_ratio_list)

if __name__ == "__main__":
    main(utility.parse_args())
