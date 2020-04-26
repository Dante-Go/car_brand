#!/bin/bash

DIRNAME=`dirname $0`
cd ../
PROJECT_NAME=`pwd`

export PYTHONPATH=$PYTHONPATH:$PROJECT_NAME
echo "PYTHONPATH="${PYTHONPATH}

python3 ${PROJECT_NAME}/AI_server/car_predict_debug.py