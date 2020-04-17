#!/bin/bash

#set the home path
DIRNAME=`dirname $0`
cd ${DIRNAME}
cd ../
AI_PROJECT_NAME=`pwd`
echo ${AI_PROJECT_NAME}
export PYTHONPATH=$PYTHONPATH:$AI_PROJECT_NAME
echo "PYTHONPATH="${PYTHONPATH}

API_SERVER=${AI_PROJECT_NAME}/API_server/api_logic_server.py
#echo ${API_SERVER}
#export PATH=$PATH:/usr/local/python3/bin
#echo "PATH="${PATH}

python3 ${API_SERVER}

