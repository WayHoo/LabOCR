# -*- coding:utf-8 -*-
import numpy as np
import validators
import cv2
import urllib.request as request
import logging as logger

def url_to_img(url):
    img = None
    try:
        resp = request.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("Get image from url failed, url: {} exception: {}".format(url, e))
    return img


def is_valid_url(string):
    return validators.url(string)
