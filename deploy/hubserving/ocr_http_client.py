import os
import imghdr
import time
import requests
import json
import base64

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF','webp','ppm'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


def main(url, image_path, batch_req=False):
    image_file_list = get_image_file_list(image_path)
    headers = {"Content-type": "application/json"}
    start_time, total_time = time.time(), 0
    images = []
    for image_file in image_file_list:
        print("loading image:{}".format(image_file))
        img = open(image_file, 'rb').read()
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        # 发送HTTP请求
        img_base64 = cv2_to_base64(img)
        images.append(img_base64)
        if not batch_req:
            data = {'images': [img_base64]}
            res = requests.post(url=url, headers=headers, data=json.dumps(data))
            print(json.dumps(res.json(), ensure_ascii=False))
    if batch_req:
        data = {'images': images}
        res = requests.post(url=url, headers=headers, data=json.dumps(data))
        print(json.dumps(res.json(), ensure_ascii=False))
    total_time = time.time() - start_time
    print("img count: %d, total time: %.2fs, avg time: %.2fs" % 
            (len(image_file_list), total_time, total_time/len(image_file_list)))


if __name__ == '__main__':
    server_url = "http://10.2.6.98:8868/predict/ocr_system"
    image_path = "./doc/test_sheets/batch_001/"
    main(server_url, image_path, batch_req=True)
