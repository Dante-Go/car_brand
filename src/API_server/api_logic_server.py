#coding=utf-8
import sys
from flask import Flask
import flask_restful
from flask_restful import request

from API_server.common.bs_define import *
from API_server.api_logic_business import *

bind_ip = '0.0.0.0'
monitorThread = None
model_base_path = ''
bind_port = PORT_API_SERVER
g_api_param_dict = None

api_url_prefix = 'api'
version_tag = 1

def get_param_dict():
    param_dict = {}
    param_dict["car_identify_ip"] = "127.0.0.1"
    param_dict["car_identify_port"] = 18888
    return param_dict


def setup_api_url(api):
    url_prefix = "%s/%s"%(api_url_prefix, version_tag)
    api.add_resource(CarPicktureQueryImpl, "/%s/car/picture_query"%(url_prefix))


def main():
    global g_api_param_dict
    g_api_param_dict = get_param_dict()
    set_global_param_dict(g_api_param_dict)
    
    app = Flask(__name__)
    api = flask_restful.Api(app)
    setup_api_url(api)
    app.run(host=bind_ip, port=bind_port, debug=True)
    
    

if __name__ == '__main__':
    main()