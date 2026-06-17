#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=0:05:00
#PJM -j
#PJM -N gpu_info

hostname
nvidia-smi
echo "===== query ====="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
