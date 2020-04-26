#coding=utf-8
import threading
import time
import sys 
import json
import base64
import io
from PIL import Image
import numpy as np
import cv2

from API_server.utils.log_util import log_print
from API_server.common.bs_define import *
from API_server.common.node_protocol import CarComparePackage
# from AI_server.car_predict import *
from AI_server.TF_slim_models import predict

class CarAnalyseServerTask(object):
    def __init__(self, client_thread, req_data, data_len):
        self.thread = client_thread
        self.req_data = req_data
        self.data_len = data_len
    def getThread(self):
        return self.thread
    def getReqData(self):
        return [self.thread, self.req_data, self.data_len]

class CarAnalyseServerWorkerThread(threading.Thread):
    def __init__(self, taskQueue, base_path, data_dir, param_dict):
        threading.Thread.__init__(self) 
        self.taskQueue = taskQueue
        self.base_path = base_path
        self.data_dir = data_dir
        self.param_dict = param_dict
        self.running = False
        
        self.model = CarPredict()
        
    def run(self):
        # log
        self.running = True
        while self.running:
            if self.taskQueue.empty():
                time.sleep(0.002)
                continue
            while self.running:
                log_print(LOG_LEVEL_DEBUG, 'Get business job in queue')
                task = self.taskQueue.get(block=False)
                (client_thread, req_data, req_data_len) = task.getReqData()
                (result, json_data) = self.doBusiness(req_data, req_data_len)
                client_thread.fill_result(action='query',processResult=result, json_data=json_data)
                if self.taskQueue.empty():
                    break
        log_print(LOG_LEVEL_DEBUG, 'CarAnalyseServerWorkerThread(%d %s) completed.'%(self.ident, self.name))
        sys.stdout.flush()
        self.stop()
    
    def stop(self):
        self.running = False
        pass
    
    def doBusiness(self, req_data, req_data_len):
        log_print(LOG_LEVEL_DEBUG, "do business starting...")
        result = 0
        request = CarComparePackage()
        success = request.parsePayload(req_data)
        if success == AI_SUCCESS:
            image = None
            if request.format == 0:
                if request.bin_len > 0:
#                     image = base64.b64decode(request.bin_data)
#                     img_array = np.fromstring(request.bin_data, np.uint8)
#                     img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
#                     cv2.imshow('img', img)
#                     cv2.waitKey()
                    
                    image = io.BytesIO(request.bin_data)
#                     img = Image.open(image)
#                     img.show()
                    car_code, car_type = self.model.car_predict(image)
#                     car_code, car_type = 1, 't'
                    print(car_code)
                    print(car_type)
                    
        info_dict_data = {}
        info_dict_data["process_result"] = AI_SUCCESS
        info_dict_data["id"] = int(car_code)
        info_dict_data["name"] = car_type
        print(info_dict_data)

        json_data = json.dumps(info_dict_data)

        return [result, json_data]
    
    def isRunning(self):
        return self.running