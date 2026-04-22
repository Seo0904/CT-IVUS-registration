#!/bin/bash
# 全エポックでバリデーション実行

EPOCHS=(best latest)
for ((i=10; i<=400; i+=10)); do
    EPOCHS+=("$i")
done

for E in "${EPOCHS[@]}"; do
    echo "==== Validating epoch: $E ===="
    bash ./WP-UNSB/run_validate_moving_mnist_paired.sh "$E"
done

echo "==== All validations finished ===="
