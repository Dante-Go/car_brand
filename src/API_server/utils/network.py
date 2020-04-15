#coding=utf-8
import time
import struct
import socket

from API_server.common.bs_define import *
from API_server.utils.log_util import log_print
from API_server.common.node_protocol import *


def net_business_communicate(ip_addr, port, cmd, busi_buf, in_buf_len):
    result = AI_SUCCESS
    sock_conn =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_conn.settimeout(30.0)
    server_addr = (ip_addr, port) 
    try:
        sock_conn.connect(server_addr)
        log_print(LOG_LEVEL_DEBUG, "Connection to %s:%d success!" % (ip_addr, port))
    except socket.timeout as e:
        log_print(LOG_LEVEL_ERROR, 'Connection to %s:%d failed' % (ip_addr, port)) 
        return [AI_FAILURE, None]
    
    n_send = 0
    n_recv = 0
    try:
        (request_buf, req_buf_len) = node_generate_request_package(cmd, busi_buf, in_buf_len)
        if request_buf != None and req_buf_len > AI_PKG_FRONT_PART_SIZE:
            n_send = net_write_package(sock_conn, request_buf, req_buf_len)
            log_print(LOG_LEVEL_DEBUG, "net_write_package() %d bytes" % (n_send))
        else:
            log_print(LOG_LEVEL_DEBUG, "Generate request error!")
            result = AI_FAILURE
    except socket.timeout:
        log_print(LOG_LEVEL_ERROR, "Send data socket timeout.")
        result = AI_FAILURE
    except socket.error as socketerror:
        log_print(LOG_LEVEL_ERROR, "Send data socket error. {0}".format(socketerror))
        result = AI_FAILURE
    
    if result == AI_SUCCESS:
        try:
            resp_pkg = None
            offset = -1
            data_recv = bytes()
            if n_send > 0:
                n_recv, data_recv, offset = net_read_package(sock_conn, data_recv)
                if n_recv > 0:
                    resp_pkg = AiNetworkPackage()
                    (cmdResp, respBusiLen, respBusiData) = resp_pkg.ParseFromString(data_recv[offset:])
                    log_print(LOG_LEVEL_DEBUG, "Response: cmd=%d, BusiDataLen=%d" % (cmdResp, respBusiLen))
                else:
                    log_print(LOG_LEVEL_DEBUG, "Receive response error")
                    result = AI_FAILURE
            else:
                log_print(LOG_LEVEL_DEBUG, "Error: Send request to %s:%d error!" %(ip_addr, port))
                result = AI_FAILURE
        except socket.timeout:
            log_print(LOG_LEVEL_ERROR, "Receive data socket timeout.")
            result = AI_FAILURE
        except socket.error as socketerror:
            log_print(LOG_LEVEL_ERROR, 'Receive to %s:%d failed' % (ip_addr, port)) 
            return [AI_FAILURE, None] 
        
    sock_conn.close()
    return [result, resp_pkg]
    
def node_check_package_buffer(data_buffer, buf_len):
    need_continue_recv = False
    offset = 0
    foundPrefix = False
#     structPrefix = struct.Struct('<I')
    while offset < buf_len:
        if (offset+4) > buf_len:
            break
        prefix, = struct.unpack('<I', data_buffer[offset:(offset+4)])
        if prefix == AI_PKG_PREFIX:
            foundPrefix = True
            break
        offset += 1
    if foundPrefix:
        if (offset + AI_PKG_FRONT_PART_SIZE) > buf_len:
            return [False, offset, need_continue_recv]
        headPack = struct.unpack('<IIII', data_buffer[offset:(offset + AI_PKG_FRONT_PART_SIZE)])
        pkgDataLen = headPack[3]
        end_offset = offset + AI_PKG_FRONT_PART_SIZE + pkgDataLen + AI_PKG_END_PART_SIZE
        if (end_offset) > buf_len:
            need_continue_recv = True
            return [False, offset, need_continue_recv]
        suffix, = struct.unpack('<I', data_buffer[(end_offset-4):end_offset])
        if suffix == AI_PKG_SUFFIX:
            return [True, offset, need_continue_recv]
        else:
            return [False, offset, need_continue_recv]
    else:
        return [False, offset, need_continue_recv]
            

def net_read_package(sock_conn, data_buffer):
    start_time = time.time()
    if sock_conn == None or data_buffer == None:
        return [-1, data_buffer, 0]
    result = -1
    total_byte = 0
    n_recv = 0
    retry = 16
    check = False
    offset = -1
    while retry > 0:
        try:
            data = sock_conn.recv(AI_TCP_BUF_MAX_LEN)
        except BlockingIOError as e:
            pass
        if not data:
            result = 0
            break
        n_recv = len(data)
        data_buffer += data 
        total_byte += n_recv
        
        if total_byte < AI_PKG_FRONT_PART_SIZE:
            retry -= 1
            continue
        
        check, offset, need_recv = node_check_package_buffer(data_buffer, total_byte)
        while need_recv:
            data_continue = None
            try:
                data_continue = sock_conn.recv(AI_TCP_BUF_MAX_LEN)
            except BlockingIOError as e:
                pass
            if not data_continue:
                log_print(LOG_LEVEL_ERROR, "Receive data error")
                result = 0
                break
            n_recv = len(data_continue)
            data_buffer += data_continue
            total_byte += n_recv
            check, offset, need_recv = node_check_package_buffer(data_buffer, total_byte)
        if check:
            result = total_byte
            break
        retry -= 1
    return [result, data_buffer, offset]


def net_write_package(sock_conn, data_buffer, data_len):
    if sock_conn == None or data_buffer == None or data_len < 1:
        return -1
    start_time = time.time()
    result = -1
    total_byte = 0
    n_send = 0
    retry = 16
    while total_byte < data_len and retry > 0:
        n_send = sock_conn.send(data_buffer[total_byte:])
        if n_send == 0:
            log_print(LOG_LEVEL_DEBUG, 'Maybe the socket has closed.')
            result = 0
            break
        if n_send < 0:
            log_print(LOG_LEVEL_DEBUG, 'write socket error.')
            break
        total_byte += n_send
        retry -= 1
    log_print(LOG_LEVEL_DEBUG, 'net_write_package() end, cost time = %f s'%(time.time() - start_time))
    if total_byte > 0:
        return total_byte
    else:
        return result
    
    