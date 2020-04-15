#coding=utf-8

AI_TCP_BUF_MAX_LEN = 1024000
AI_PKG_FRONT_PART_SIZE = 16
AI_PKG_END_PART_SIZE = 8

AI_PKG_PREFIX = 0xFAFAFAFA
AI_PKG_SUFFIX = 0xFBFBFBFB

AI_PKG_MAGIC = 0x00000000

API_CMD_HEART_BEAT = 2
API_CMD_CAR_ANALYSE = 5
API_CMD_CAR_BRAND_QUERY = 3

PORT_API_SERVER = 12100

LOG_LEVEL_ERROR = 1
LOG_LEVEL_WARN = 2
LOG_LEVEL_INFO = 3
LOG_LEVEL_DEBUG = 4


AI_RESPONSE_BIT = 0x00010000


AI_SUCCESS = 0
AI_FAILURE = -1


