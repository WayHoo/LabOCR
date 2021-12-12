# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time
import cv2
import json

sys.path.insert(0, ".")

import logging as logger
from paddlehub.module.module import moduleinfo, serving
import paddlehub as hub
from tools.infer.utility import base64_to_cv2
from tools.infer.predict_system import TextSystem
from kie.tsr.table_rec import TableRecognizer
from tools.infer.utility import parse_args
from deploy.hubserving.ocr_system.params import read_params
from deploy.hubserving.utils.image import is_valid_url, url_to_img


@moduleinfo(
    name="ocr_system",
    version="1.0.0",
    summary="ocr system service",
    author="WayHoo",
    author_email="huwei.debug@foxmail.com",
    type="cv/text_recognition")
class OCRSystem(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        cfg = self.merge_configs()

        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn

        self.text_sys = TextSystem(cfg)
        self.table_rec = TableRecognizer(cfg)

    def merge_configs(self, ):
        # deafult cfg
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        cfg = parse_args()

        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.error("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def predict(self, images=[], paths=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of chinese texts and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        all_results = []
        for img in predicted_data:
            if img is None:
                logger.error("error in loading image")
                all_results.append([])
                continue
            start_time = time.time()
            dt_boxes, rec_res = self.text_sys(img)
            elapse = time.time() - start_time
            logger.info("Predict time: {}".format(elapse))

            sheets, entries, _ = self.table_rec(img, "", dt_boxes, rec_res, save=False)
            all_results.append({"sheets": sheets, "entries": entries})
            # dt_num = len(dt_boxes)
            # rec_res_final = []

            # for dno in range(dt_num):
            #     text, score = rec_res[dno]
            #     rec_res_final.append({
            #         'text': text,
            #         'score': float(score),
            #         'box': dt_boxes[dno].astype(np.int).tolist()
            #     })
            # all_results.append(rec_res_final)
        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        :param: images: maybe the base64 format of image or image urls, each item is type of str
        """
        images_decode = [url_to_img(s) if is_valid_url(s) else base64_to_cv2(s) for s in images]
        results = self.predict(images_decode, **kwargs)
        return results


if __name__ == '__main__':
    ocr = OCRSystem()
    ocr._initialize(use_gpu=True, enable_mkldnn=True)
    # image_path = [
    #     './doc/test_sheets/batch_001/test_sheet (1).jpg',
    #     './doc/test_sheets/batch_001/test_sheet (2).jpg',
    # ]
    # res = ocr.predict(paths=image_path)
    img_urls = ["http://p5.itc.cn/images01/20201009/5c56e82e019f4e169c7f802f57484b9b.jpeg", 
                "http://s04.lmbang.com/M00/D2/81/tzlM81Vum9uANyzoAAPly4Oz-KQ997.jpg%21r750x600x100.jpg", 
                "https://iknow-pic.cdn.bcebos.com/0eb30f2442a7d93323662f85ad4bd11372f0017e"]
    images_decode = [url_to_img(s) if is_valid_url(s) else base64_to_cv2(s) for s in img_urls]
    # cv2.imwrite("/home/huwei/Projects/PaddleOCR/output/test_sheet.jpg", images_decode[0])
    res = ocr.predict(images=images_decode)
    print(json.dumps(res, ensure_ascii=False))
