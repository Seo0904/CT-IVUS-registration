#!/bin/bash
docker run \
    -d \
    --init \
    -p 15002:5000 \
    -p 16008:6006 \
    -p 18521-18531:8501-8511 \
    -p 18889:8888 \
    -it \
    --gpus=all \
    --ipc=host \
    --name=ct_ivus_registration \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=/mnt/storage/CT-IVUS_registration:/workspace/data \
    ct_ivus_registration:latest \
    ${@-fish}
