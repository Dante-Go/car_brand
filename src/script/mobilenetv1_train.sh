#!/bin/bash

#set the home path
DIRNAME=`dirname $0`
cd ${DIRNAME}
cd ../
AI_PROJECT_NAME=`pwd`
echo ${AI_PROJECT_NAME}
export PYTHONPATH=$PYTHONPATH:$AI_PROJECT_NAME
echo "PYTHONPATH="${PYTHONPATH}

TRAIN=${AI_PROJECT_NAME}/AI_server/TF_slim_models/mobilenet_train_val.py

python3 ${TRAIN}
