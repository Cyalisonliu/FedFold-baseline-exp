# FedFold

## How to run

Use `train_fedfold.sh`.

`quant_bits` is set to `-1` for random quantization level.

## Configuration

1. Change to different size of ResNet:

In `model.py` , ResNet class, change `num_blocks = [.,.,.,.]`

```bash
ResNet18 -> [2,2,2,2]
ResNet152 -> [3,8,36,3]
```

2. Change to different models:

In `data.py`, `set_parameters` function, change `model_fn` in the dataset that yo train.