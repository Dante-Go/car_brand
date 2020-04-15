#coding=utf-8

from API_server.common.bs_define import *

G_LOG_LEVEL_FLAG = LOG_LEVEL_DEBUG

def log_print(level, text):
    if G_LOG_LEVEL_FLAG < level:
        return
    if LOG_LEVEL_ERROR == level:
        print("[Error] %s" % (text)) 
    elif LOG_LEVEL_WARN == level:
        print("[Warn] %s" % (text))
    elif LOG_LEVEL_INFO == level:
        print("[Info] %s" % (text))
    elif LOG_LEVEL_DEBUG == level:
        print("[Debug] %s" % (text))
