#!/bin/bash

# 10エポックごとにvalを実行
for epoch in $(seq 10 10 200); do
    ./run_validate_moving_mnist_paired.sh $epoch
done
