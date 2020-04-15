#coding=utf-8
import flask_restful
import time
import base64
from flask_restful import request, reqparse

from API_server.api_logic_extend import *

g_api_param_dict = None
g_token_map = {}

access_parser = reqparse.RequestParser()

def set_global_param_dict(param_dict):
    global g_api_param_dict
    g_api_param_dict = param_dict

def validate_token(token_map, token):
    result = extend_validate_token(g_api_param_dict, token_map, token)
    return result

class CarPicktureQueryImpl(flask_restful.Resource):
    def _car_picture_query(self, picture, pic_file_name, image_format):
        success, data_dict = False, None
        executor = CarPictureQueryExecutor(param_dict=g_api_param_dict)
        if picture is not None:
            picture_bin = base64.b64decode(picture)
            success, data_dict = executor.query_car_by_picture(picture_bin, image_format)
        elif pic_file_name is not None:
            success, data_dict = executor.query_car_by_filename(pic_file_name)
        return success, data_dict
    
    def _check_token(self, token):
        return validate_token(g_token_map, token)
    
    def _request_process(self, args):
        cost_start = time.time()
        
#         access_token = args["access_tocken"]
#         if access_token is None:
#             return {"result":"fail", "error":"access_token is None"}, 200
#         if self._check_token(access_token) == False:
#             return {"result":"fail", "error":"access_token is invalid"}, 200
        
        json_data = request.get_json(force=True)
        if json_data is None:
            return {"result":"fail", "error":"No json data. Only support application/json Content-Type."}, 200
        
        start_time = None
        end_time = None
        picture = None
        pic_file_name = None
        image_format = None
        
        if "picture" in json_data:
            picture = json_data["picture"]
        if "pic_file_name" in json_data:
            pic_file_name = json_data["pic_file_name"]
        if "image_format" in json_data:
            image_format = json_data["image_format"]
            
        if picture is None and pic_file_name is None:
            return {"result":"fail", "error":"picture and pic_file_name can not empty both."}, 200
        
        success, data_dict = self._car_picture_query(picture, pic_file_name, image_format)
        if success == True:
            cost_time = time.time() - cost_start
            if data_dict is not None:
                data_dict['cost_time'] = cost_time
            return data_dict, 200
        else:
            if data_dict is None:
                data_dict = {"result":"fail", "error":"Inner error."}
            return data_dict, 200
        
    
    def post(self):
        s_time = time.time()
        args = access_parser.parse_args()
        data_dict, http_code = self._request_process(args)
        cost_time = time.time() -s_time
        return data_dict, http_code
    
    def get(self):
        return {"result":"fail", "error":"Not suport GET method"}