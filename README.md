# HDRNN
The Pytoch implementation of ``Hybrid Directional Graph Neural Network with Learnable Equivariance''. This code is based on [OCP](https://github.com/Open-Catalyst-Project/ocp/tree/main). The solutions of issues on installation or dataset may be found in OCP benchmark.

## Installation
See [installation instructions](https://github.com/AnonymousACode/HDGNN/blob/main/INSTALL.md).

## Download data
Dataset download links and instructions are in [DATASET.md](https://github.com/AnonymousACode/HDGNN/blob/main/DATASET.md).

## Training
To train a HDGNN model for the IS2RE task, run:
```
python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 8 --mode train --config-yml configs/is2re/all/hdgnn.yml
```
Next, run this model on the test data:
```
python main.py --mode predict --config-yml configs/is2re/all/hdgnn.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```