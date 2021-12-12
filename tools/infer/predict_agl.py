import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import cv2
import numpy as np
import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list

logger = get_logger()


class AngleDetector(object):
    def __init__(self, args):
        self.args = args
        self.angle = [0, 90, 180, 270]
        angle_model_pb = os.path.join(args.agl_det_model_dir, "Angle-model.pb")
        angle_model_pbtxt = os.path.join(args.agl_det_model_dir, "Angle-model.pbtxt")
        self.angle_net = cv2.dnn.readNetFromTensorflow(angle_model_pb, angle_model_pbtxt)

    def detect_angle(self, img):
        h, w = img.shape[:2]
        if self.args.use_size_adjust:
            # cut the edge of image
            thresh = self.args.adjust_thresh
            x_min, y_min = int(thresh * w), int(thresh * h)
            x_max, y_max = w - x_min, h - y_min
            img = img[y_min: y_max, x_min: x_max]
        input_blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(224, 224), swapRB=True,
                                           mean=[103.939, 116.779, 123.68], crop=False)
        self.angle_net.setInput(input_blob)
        pred = self.angle_net.forward()
        index = np.argmax(pred, axis=1)[0]
        return self.angle[index]

    def __call__(self, img):
        img = np.array(img)
        angle = self.detect_angle(np.copy(img))
        if angle == 90:
            img = cv2.transpose(img)
            img = cv2.flip(img, flipCode=0)  # counter clock wise
        elif angle == 180:
            img = cv2.flip(img, flipCode=-1)  # flip the image both horizontally and vertically
        elif angle == 270:
            img = cv2.transpose(img)
            img = cv2.flip(img, flipCode=1)  # clock wise rotation
        logger.info("angle detected: {}°".format(angle))
        return angle, img


if __name__ == "__main__":
    args = utility.parse_args()
    args.image_dir = "./train_data/TestSheetDetEval/0/Batch01/"
    image_file_list = get_image_file_list(args.image_dir)
    angle_detector = AngleDetector(args)
    draw_img_save = "./inference_results/angle/"
    visualize = False
    if visualize and not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    rotate_dict = {-1: ((90, 270), 0), cv2.ROTATE_90_CLOCKWISE: ((0, 180), 90),
                   cv2.ROTATE_180: ((90, 270), 180),
                   cv2.ROTATE_90_COUNTERCLOCKWISE: ((0, 180), 270)}
    # opti_err_num表示可以优化的错误检测数量，acc_err_num表示严格的误识别数量
    # 0度与180度、90度与270度之间的误检，可以优化
    total, opti_err_num, acc_err_num = 0, 0, 0
    begin = time.time()
    targets = dict()
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        # 对正向图片旋转指定角度，再测试准确率
        rotate_type = cv2.ROTATE_90_CLOCKWISE
        if rotate_type != -1:
            img = cv2.rotate(img, rotate_type)
        if img is None:
            continue
        total += 1
        angle, agl_img = angle_detector(img)
        if angle == 90:
            targets[image_file] = angle
        if angle == 180:
            targets[image_file] = angle
        if angle == 270:
            targets[image_file] = angle
        if angle in rotate_dict[rotate_type][0]:
            opti_err_num += 1
        if angle != rotate_dict[rotate_type][1]:
            acc_err_num += 1
        logger.info("Angle predicted of {}: {}°".format(image_file, angle))
        if visualize:
            img_path = os.path.join(draw_img_save,
                    "agl_det_{}".format(os.path.basename(image_file)))
            cv2.imwrite(img_path, agl_img)
            logger.info("The visualized image saved in {}".format(img_path))
    print(targets)
    elapse = time.time() - begin
    logger.info("Angle predict, total: {}, acc_err_num: {}, opti_err_num: {}".format(total, acc_err_num, opti_err_num))
    logger.info("Real accuracy: {:.2f}%".format((total - acc_err_num) * 100 / total))
    logger.info("Optimize accuracy: {:.2f}%".format((total - opti_err_num) * 100 / total))
    logger.info("Angle predict total time: {:.2f}s".format(elapse))
    logger.info("Angle predict avg time: {:.2f}s".format(elapse/total))
    
