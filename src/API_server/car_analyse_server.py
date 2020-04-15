#coding=utf-8
import queue
import socket
import time
import signal
import select
import sys
import traceback

from API_server.logic_business import *
from API_server.utils.network import net_read_package
from API_server.utils.network import net_write_package
from API_server.common.node_protocol import *
from API_server.common.bs_define import *
from API_server.utils.log_util import log_print

busiWorker = None
bind_ip = '0.0.0.0'
bind_port = 18888

class CarAnalyseServerClientThread(threading.Thread):
    def __init__(self, client_socket, queue):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.action = 'query'
        self.json_data = None
        self.result = 0
        self.queue = queue
        self.completed = False
        self.sleep_counter = 1000 # 2s = 1000*2ms
     
    def run(self):
        log_print(LOG_LEVEL_DEBUG, 'CarAnalyseServerClinetThread(%d %s) handle client connetion start.' %(self.ident, self.name))
        service_start_time = time.time()
        data_recv = bytes()
        n_recv, data_recv, offset = net_read_package(self.client_socket, data_recv)
        if n_recv > 0:
            req_start = time.time()
            req_pkg = AiNetworkPackage()
            (cmdReq, reqBusiLen, reqBusiData) = req_pkg.ParseFromString(data_recv[offset:])
            log_print(LOG_LEVEL_DEBUG, 'Request: cmd=%d, BusiDataLen=%d' % (cmdReq, reqBusiLen))
            log_print(LOG_LEVEL_DEBUG, 'request analyse cost time=%f s' %(time.time() - req_start))
            if cmdReq == API_CMD_HEART_BEAT:
                log_print(LOG_LEVEL_DEBUG, 'Heart beat request process')
                sys.stdout.flush()
                timestamp = int(round(time.time()))
                load_score = 50
                resp_pkg = HeartBeatPackage()
                (resp_buf, resp_buf_len) = resp_pkg.createResponse(flag=0, status=1, capacity=1.0, listen_port=bind_port, 
                                                                   api_cmd=API_CMD_CAR_ANALYSE, load_score=load_score, 
                                                                   ip=bind_ip, timestamp=timestamp)
             
            else:
                log_print(LOG_LEVEL_DEBUG, 'Business request process')
                task = CarAnalyseServerTask(self, reqBusiData, reqBusiLen)
                log_print(LOG_LEVEL_DEBUG, 'Now create a task...')
                self.queue.put(task)
                 
                start_time = time.time()
                while self.sleep_counter > 0:
                    if self.completed == True:
                        break
                    time.sleep(0.001)
                    self.sleep_counter -= 1
                cost_time = int(round(time.time() - start_time) * 1000)
                 
                if self.completed == False:
                    self.result = 1
                    log_print(LOG_LEVEL_DEBUG, 'Business process timeout')
                 
                byte_start = time.time()
                resp_pkg = CarComparePackage()
                (resp_buf, resp_buf_len) = resp_pkg.createResponse(jsonData=self.json_data, processResult=self.result, costTime=cost_time)
                cur_time = time.time()
                log_print(LOG_LEVEL_DEBUG, 'Construct response cost time = %f s' % (cur_time - byte_start))
                 
            transmit_start = time.time()
            resp_pkg_buf, resp_pkg_len = node_generate_response_package(cmdReq, resp_buf, resp_buf_len)
            n_send = net_write_package(self.client_socket, resp_pkg_buf, resp_pkg_len)
             
            if n_send > 0:
                log_print(LOG_LEVEL_DEBUG, 'Send response ok. cost time = %f s' % (time.time() - transmit_start))
            else:
                log_print(LOG_LEVEL_DEBUG, 'Send response failed. cost time = %f s' % (time.time() - transmit_start))
        else:
            log_print(LOG_LEVEL_DEBUG, 'Receive request error')
             
        self.client_socket.close()
         
        service_end_time = time.time()
        log_print(LOG_LEVEL_DEBUG, 'service process cost time = %f s' %(service_end_time - service_start_time))
        sys.stdout.flush()
     
    def fill_result(self, action, processResult, json_data):
        self.action = action
        self.result = processResult
        self.json_data = json_data
        self.completed = True
     
     
class MonitorThread(threading.Thread):
    def __init__(self, busi_thread, taskQueue):
        threading.Thread.__init__(self)
        self.busi_thread = busi_thread
        self.taskQueue = taskQueue
        self.running = False
     
    def run(self):
        log_print(LOG_LEVEL_DEBUG, 'MonitorThread(%d %s) handle client connection start.'%(self.ident, self.name))
        self.runing = True
        while self.running == True:
            time.sleep(100.0)
            if self.busi_thread is None or self.busi_thread.isRunning() == False:
                self.busi_thread = restart_business_thread(self.taskQueue)
        self.running = False
        log_print(LOG_LEVEL_DEBUG, 'MonitorThread is terminate')
    def stop(self):
        self.running = False
         
 
def start_server_listen():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(False)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_ip, bind_port))
    server.listen(5)
    log_print(LOG_LEVEL_DEBUG, 'Listening on {}:{}'.format(bind_ip, bind_port))
    return server 
 
def start_business_thread(taskQueue):
    model_base_path = ''
    data_dir = ''
    param_dict = {}
    param_dict['model_base_path'] = model_base_path
    param_dict['data_dir'] = data_dir
    worker = CarAnalyseServerWorkerThread(taskQueue, model_base_path, data_dir, param_dict)
    worker.start()
    return worker 
 
def signal_handler(signum):
    log_print(LOG_LEVEL_DEBUG, 'Received signal: {0}'.format(signum))
    time.sleep(1)
    busiWorker.stop()
    time.sleep(5)
    exit()
     
def restart_business_thread(taskQueue):
    global busiWorker
    try:
        if busiWorker is not None:
            busiWorker.stop()
    except Exception as ee:
        exstr = traceback.format_exc()
        print(exstr)
    busiWorker = start_business_thread(taskQueue)
    return busiWorker


def main():
    global busiWorker
    LISTEN_LIST = []
    serverSocket = start_server_listen()
    LISTEN_LIST.append(serverSocket)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
     
    taskQueue = queue.Queue(8)
    busiWorker = start_business_thread(taskQueue)
     
#     monitorThread = MonitorThread(busiWorker, taskQueue)
#     monitorThread.start()
         
    mainRunning = True
    timeout = 2
    while mainRunning:
        read_sockets, write_sockets, error_sockets = select.select(LISTEN_LIST, [], [], timeout)
        for sock in read_sockets:
            if sock == serverSocket:
                client_sock, address = serverSocket.accept()
                print('Accepted connectin from {}:{}'.format(address[0], address[1]))
                client_thread = CarAnalyseServerClientThread(client_sock, taskQueue)
                client_thread.start()
                sys.stdout.flush()
     
    serverSocket.shutdown(socket.SHUT_RDWR)
    taskQueue.join()

if __name__ == '__main__':
    main()