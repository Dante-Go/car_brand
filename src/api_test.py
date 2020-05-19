#coding=utf-8
import os
import requests
import json
import base64
import cv2
import numpy as np
import io
from PIL import Image

import time

api_url = 'http://10.99.16.52:12100/api/1/car/picture_query'
# img_path = '/data/car_brand_model/api_test_img2.jpg'
# img_path = '/data/car_brand_model/data/train/丰田/20200104205208_out_v_1_粤SL2C01_noBG.jpg'
img_path_base = '/data/car_brand_model/data/val_58/data/'

test_log = '/data/car_brand_model/data/val_58/test.txt'
f_log = open(test_log, 'w')
for root, dirs, files in os.walk(img_path_base):
    for file in files:
        img_path = os.path.join(root, file)
        
        f = open(img_path, 'rb')
        base64_data = base64.b64encode(f.read())
        base64_str = base64_data.decode()
        # print(base64_str)
         
        # image = base64.b64decode(base64_str)
        # img_array = np.fromstring(image, np.uint8)
        # img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # 
        # image = io.BytesIO(image)
        # img = Image.open(image)
        # img.show()
         
        data = {"picture":base64_str, "image_format":"jpg" }
#         print(base64_str)
        json_data = json.dumps(data)
          
        headers = {"Content-Type":"application/json"}
          
        resp = requests.post(api_url, headers=headers, data=json_data)
        print(img_path)
        print(resp.text)
        str = '{}\n {}\n\n'.format(img_path, resp.text)
        f_log.write(str)
        time.sleep(1)
        # img = Image.open(img_path)
        # print(img.size)
        # (width, height) = img.size
        # region = (0, height*0.4, width, height*0.85)
        # img_crop = img.crop(region)
        # # img.show()
        # img_crop.show()
f_log.close()
