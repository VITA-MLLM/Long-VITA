#!/bin/bash

set -e
set -x

timestamp=`date +'%Y%m%d_%H%M%S'`

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

######################################################################

export LongVITA_URL=http://172.17.0.242:5001/api

cd ${LOCAL_CODE_PATH}

curl -X PUT ${LongVITA_URL} \
	-H "Content-Type: application/json" \
	-d '{
		"prompts": ["San Francisco is a"],
		"image_path_list": [],
		"video_path_list": []
        }'

set +x
