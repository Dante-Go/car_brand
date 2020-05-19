#!/bin/bash

#set the home path
DIRNAME=`dirname $0`
cd ${DIRNAME}
cd ../
AI_PROJECT_NAME=`pwd`
echo ${AI_PROJECT_NAME}
export PYTHONPATH=$PYTHONPATH:$AI_PROJECT_NAME
echo "PYTHONPATH="${PYTHONPATH}

CREATE_LABELS=${AI_PROJECT_NAME}/AI_server/TF_slim_models/create_labels_files.py

CREATE_TFRECORDS=${AI_PROJECT_NAME}/AI_server/TF_slim_models/create_tf_record.py

python3 ${CREATE_LABELS}
python3 ${CREATE_TFRECORDS}
