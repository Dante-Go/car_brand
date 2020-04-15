#coding=utf-8
import time
import traceback
import json

from API_server.common.bs_define import *
from API_server.utils.log_util import log_print
from API_server.common.node_protocol import *
from API_server.utils.network import *

class CarPictureQueryExecutor(object):
    def __init__(self, param_dict=None):
        self.param_dict = param_dict
        self.car_identify_ip = param_dict["car_identify_ip"]
        self.car_identify_port = param_dict["car_identify_port"]
        
    def query_car_by_picture(self, picture, image_format):
        success, data_dict = False, None
        success, car_info = self.do_car_identify(picture)
        if success == False:
            return success, {"result":"fail", "error":"catch error when identify car picture."}
        if car_info is None:
            return success, {"result":"fail", "message":"car info error"}
        car_brand_id, car_brand_name = car_info
        result_dict = {}
        result_dict["car_brand_id"] = car_brand_id
        result_dict["car_brand_name"] = car_brand_name
        data_dict = {"result":"success", "data":result_dict}
        return success, data_dict
    
    def query_car_by_filename(self, pic_file_name):
        pass

    def do_car_identify(self, jpeg_image):
        ret = False
        car_info = None
        start_time = time.time()
        success = False
        cmd = API_CMD_CAR_BRAND_QUERY
        
        try:
            format = 0
            imageWidth = 0
            imageHeight = 0
            jsonData = "{}"
            binData = jpeg_image
            binLen = len(binData)
            car_pkg = CarComparePackage()
            req_buf, req_buf_len = car_pkg.createRequest(format=format, imageWidth=imageWidth, imageHeight=imageHeight, binLen=binLen, binData=binData, 
                                                         jsonData=jsonData, processResult=0, score=0.0, costTime=0)
            (result, resp_pkg) = net_business_communicate(self.car_identify_ip, self.car_identify_port, cmd, req_buf, req_buf_len)
            network_time = time.time()
            if result == AI_SUCCESS:
                (respCmd, resp_busi_len, resp_busi_buf) = resp_pkg.GetData()
                success = car_pkg.parsePayload(resp_busi_buf)
                if success == AI_SUCCESS:
                    json_dict = json.loads(car_pkg.json_data)
                    if json_dict["process_result"] == AI_SUCCESS:
                        car_brand_id = json_dict["id"]
                        car_brand_name = json_dict["name"]
                        car_info = [car_brand_id, car_brand_name]
            ret = True
        except Exception as ee:
            exstr = traceback.format_exc()
            print(exstr)
        log_print(LOG_LEVEL_DEBUG, "Car identify analyse cost %f s."%(time.time()-start_time))
        return ret, car_info
    
    

def extend_validate_token(g_api_param_dict, token_map, token):
    result = False
    
    now_time = time.time()
    token_expire_time = g_api_param_dict["token_expire_time"]
    if token in token_map:
        client_id, logon_time, expire_time = token_map[token] 
        if expire_time < now_time:
            token_map.pop(token)
        else:
            expire_time = int(time.time() + token_expire_time)
            token_map[token] = [client_id, logon_time, expire_time]
            result = True
    
    return result

