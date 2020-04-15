#coding=utf-8
import struct
import ctypes
import json
import binascii
import traceback

from API_server.common.bs_define import *
from API_server.utils.log_util import log_print

def node_generate_request_package(cmd, busi_buf, in_buf_len):
    if not busi_buf:
        log_print(LOG_LEVEL_DEBUG, "Request business data buffer is None")
        return None
    pkg = AiNetworkPackage(cmd=cmd, dataLen=in_buf_len, pkgData=busi_buf)
    (out_pkg_buf, out_pkg_len) = pkg.SerializeToString()
    return [out_pkg_buf, out_pkg_len]

def node_generate_response_package(cmd, busi_buf, in_buf_len):
    if not busi_buf:
        return None
    cmdResp = cmd | AI_RESPONSE_BIT
    pkg = AiNetworkPackage(cmd=cmdResp, dataLen=in_buf_len, pkgData=busi_buf)
    (out_pkg_buf, out_pkg_len) = pkg.SerializeToString()
    return [out_pkg_buf, out_pkg_len]

# class CarAnalysePackage(object):
#     def __init__(self, action='query', jsonData=None, processResult=0, costTime=0):
#         self.action = action
#         self.img_format = 'jpg'
#         self.process_result = processResult
#         self.cost_time = costTime
#         self.json_data = jsonData
#     
#     def constructPayload(self, direction=0, action='query', jsonData=None, processResult=0, costTime=0):
#         dict_data = {}
#         
#         json_str_data = json.dumps(dict_data)
#         payload_data = binascii.b2a_hex(json_str_data.encode(encoding='utf_8'))
#         data_len = len(payload_data)
#         
#         payload = AiCommonPayload(direction=direction, dataLen=data_len, payload=payload_data)
#         return payload.SerializeToString()
#     
#     def createResponse(self, action='query', jsonData=None, processResult=0, costTime=0):
#         return self.constructPayload(direction=1, action=action, jsonData=jsonData, processResult=processResult, costTime=costTime)
    

class CarComparePackage(object):
    def __init__(self, format=0, imageWidth=0, imageHeight=0, binLen=0, binData=None, jsonData=None, processResult=0, score=0.0, costTime=0):
        self.format = format
        self.image_width = imageWidth
        self.image_height = imageHeight
        self.process_result = processResult
        self.score = score
        self.json_data = jsonData
        self.bin_len = binLen
        self.bin_data = binData
        
    def parsePayload(self, payload_buf):
        result = AI_FAILURE
        try:
            payload = AiCommonPayload()
            (self.direction, data_len, bin_len, data, bin_data) = payload.ParseFromString(payload_buf)
            if data_len > 0:
                data = binascii.a2b_hex(data) 
                log_print(LOG_LEVEL_DEBUG, "Car Analyse: Parse payload, data={0}.".format(data))
                dict_data = json.loads(data)
                self.format = dict_data["format"]
                self.image_width = dict_data["image_width"]
                self.image_height = dict_data["image_height"]
                self.process_result = dict_data["process_result"]
                self.score = dict_data["score"]
                self.cost_time = dict_data["cost_time"]
                self.json_data = dict_data["json_data"]
                self.bin_len = bin_len
                self.bin_data = bin_data
                self.data_normal = True
            result = AI_SUCCESS
        except Exception as ee:
            result = AI_FAILURE
            exstr = traceback.format_exc()
            print(exstr)
        return result
        
    def constructPayload(self, direction=0, format=0, imageWidth=0, imageHeight=0, binLen=0, binData=None, jsonData=None, processResult=0, score=0.0, costTime=0):
        dict_data = {}
        dict_data["format"] = format 
        dict_data["image_width"] = imageWidth
        dict_data["image_height"] = imageHeight
        dict_data["json_data"] = jsonData
        dict_data["process_result"] = processResult
        dict_data["score"] = score
        dict_data["cost_time"] = costTime
        
        json_str_data = json.dumps(dict_data) 
        payload_data = binascii.b2a_hex(json_str_data.encode(encoding='utf-8'))
        data_len = len(payload_data)
        payload = AiCommonPayload(direction=direction, dataLen=data_len, payload=payload_data, binLen=binLen, binData=binData)
        return payload.SerializeToString()
    
    def createRequest(self, format=0, imageWidth=0, imageHeight=0, binLen=0, binData=None, jsonData=None, processResult=0, score=0.0, costTime=0):
        return self.constructPayload(direction=0, format=format, imageWidth=imageWidth, imageHeight=imageHeight, 
                                     binLen=binLen, binData=binData, jsonData=jsonData, processResult=processResult, score=score, costTime=costTime)
        
    def createResponse(self, format=0, imageWidth=0, imageHeight=0, binLen=0, binData=None, jsonData=None, processResult=0, score=0.0, costTime=0):
        return self.constructPayload(direction=1, format=format, imageWidth=imageWidth, imageHeight=imageHeight, 
                                     binLen=binLen, binData=binData, jsonData=jsonData, processResult=processResult, score=score, costTime=costTime)


class AiNetworkPackage(object):
    def __init__(self, cmd=0, dataLen=0, pkgData=None):
        self.prefix = AI_PKG_PREFIX
        self.magic = AI_PKG_MAGIC
        self.command = cmd 
        self.length = dataLen
        self.data = pkgData
        self.crc = 0
        self.suffix = AI_PKG_SUFFIX
        self.default_format = '<IIIIII'
        self.front_part_format = '<IIII'
        self.front_part_size = AI_PKG_FRONT_PART_SIZE
        self.end_part_size = AI_PKG_END_PART_SIZE
        if dataLen > 0:
            self.struct = struct.Struct('<IIII%dsII' % (dataLen))
        else:
            self.struct = struct.Struct(self.default_format)
        
    def SerializeToString(self):
        if self.length > 0:
            values = (self.prefix, self.magic, self.command, self.length, self.data, self.crc, self.suffix)
        else:
            values = (self.prefix, self.magic, self.command, self.length, self.crc, self.suffix)
        bufSize = self.front_part_size + self.end_part_size + self.length
        buffer = ctypes.create_string_buffer(bufSize)
        self.struct.pack_into(buffer, 0, *values)
        return [buffer.raw, bufSize]
        
    def ParseFromString(self, data):
        if not data:
            log_print(LOG_LEVEL_DEBUG, 'Data is None')
            return None
        structHeader = struct.Struct(self.front_part_format)
        (self.prefix, self.magic, self.command, self.length) = structHeader.unpack_from(data[:structHeader.size])
        if self.length > 0:
            self.data = data[structHeader.size:(self.length + structHeader.size)]
        
        return [self.command, self.length, self.data]
    
    def GetData(self):
        return [self.command, self.length, self.data]


class AiCommonPayload(object):
    def __init__(self, direction=0, dataLen=0, payload=None, binLen=0, binData=None):
        self.direction = direction
        self.length = dataLen
        self.binlen = binLen
        self.payload = payload
        self.bindata = binData
        self.default_format = '<III'
        self.struct = struct.Struct(self.default_format)
        if dataLen > 0 and binLen > 0:
            self.struct = struct.Struct('<III%ds%ds' % (dataLen, binLen))
        elif dataLen > 0 and binLen == 0:
            self.struct = struct.Struct('<III%ds' % (dataLen))
        elif dataLen == 0 and binLen > 0:
            self.struct = struct.Struct('<III%ds' % (binLen))
        else:
            self.struct = struct.Struct(self.default_format)
        
    def SerializeToString(self):
        if self.length > 0 and self.binlen > 0:
            values = (self.direction, self.length, self.binlen, self.payload, self.bindata)
        elif self.length > 0 and self.binlen == 0:
            values = (self.direction, self.length, self.binlen, self.payload)
        elif self.length == 0 and self.binlen > 0:
            values = (self.direction, self.length, self.binlen, self.bindata)
        else:
            values = (self.direction, self.length, self.binlen)
        
        bufSize = self.struct.size
        buffer = ctypes.create_string_buffer(bufSize)
        self.struct.pack_into(buffer, 0, *values)
        return [buffer.raw, bufSize]
    
    def ParseFromString(self, data):
        if not data:
            log_print(LOG_LEVEL_DEBUG, 'Data is None')
            return None
        structHeader = struct.Struct(self.default_format)
        (self.direction, self.length, self.binlen) = structHeader.unpack_from(data[:structHeader.size])
        if self.length > 0:
            self.payload = data[structHeader.size: (self.length + structHeader.size)]
        if self.binlen > 0:
            self.bindata = data[self.length + structHeader.size: (self.length + self.binlen + structHeader.size)]
        return [self.direction, self.length, self.binlen, self.payload, self.bindata]

     
class HeartBeatPackage(object):
    def __init__(self, flag=0, status=0, capacity=0, listen_port=0, 
                 api_cmd=0, load_score=0, ip=None, timestamp=0, busi_flag=0):
        self.flag = flag 
        self.status = status
        self.capacity = capacity
        self.listen_port = listen_port
        self.api_cmd = api_cmd 
        self.load_score = load_score
        self.ip = ip
        self.timestamp = timestamp
        self.busi_flag = busi_flag
        self.data_normal = False
    
    def parsePayload(self, payload_buf):
        pass
    def constructPayload(self, direction=0,flag=0, status=0, capacity=0, listen_port=0, api_cmd=0, load_score=0, ip=None, timestamp=0, busi_flag=0):
        dict_data = {}
        dict_data['flag'] = flag
        dict_data['status'] = status
        dict_data['capacity'] = capacity
        dict_data['listen_port'] = listen_port
        dict_data['api_cmd'] = api_cmd
        dict_data['load_score'] = load_score
        dict_data['ip'] = ip
        dict_data['timestamp'] = timestamp
        dict_data['busi_flag'] = busi_flag
        json_str_data = json.dumps(dict_data)
        payload_data = binascii.b2a_hex(json_str_data.encode(encoding='utf_8'))
        data_len = len(payload_data)
        payload = AiCommonPayload(direction=direction, dataLen=data_len, payload=payload_data)
        return payload.SerializeToString()
        
    def createRequest(self):
        pass
    def createResponse(self, flag=0, status=0, capacity=0, listen_port=0, api_cmd=0, load_score=0, ip=None, timestamp=0, busi_flag=0):
        return self.constructPayload(direction=1, flag=flag, status=status, capacity=capacity, listen_port=listen_port, api_cmd=api_cmd, load_score=load_score, ip=ip, timestamp=timestamp, busi_flag=busi_flag)
    
        