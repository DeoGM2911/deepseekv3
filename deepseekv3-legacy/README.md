This code is written when I first try to just read the paper and there are some part that is incorrect due to misunderstanding. The correct implementation and documentation is sotred under the other deepseekv3.2 directory.

# DeepSeekV3 Architecture Reconstruction Project

This project aims to reconstruct the DeepSeekV3 architecture based on my understanding of the papers. This repo is for learning purposes only.

## Requirements

- Python 3.12
- PyTorch 2.2.0
- transformers 4.36.2
- datasets 2.16.3

Run `pip install -r requirements.txt` to install the requirements.

## Model

The model is implemented in [model/deepseek.py](./model/deepseek.py). For further details about the architecture, please refer to the [model/README.md](./model/README.md).

## Training

The training code is implemented in [train.py](./train.py). I also go ahead and try to implement the GRPO algorithm for RLHF in [grpo_algo.py](./finetune/grpo_algo.py). Nevertheless, in the real world, we would use external libraries like `trl` to finetune.

I also implememt the MoE's router loss in [moe_loss.py](./loss/moe_loss.py).