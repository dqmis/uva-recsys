# RecSys Challenge 2024 Reproducibility Code

## Further investigating transformer models in recommender systems

This repo contains the code for the RecSys Challenge 2024 Recommender Systems course project. The project aims to investigate the performance of transformer models in recommender systems. This is an extension of the work done by [Moreira et. al.](https://scontent.fein1-1.fna.fbcdn.net/v/t39.8562-6/246721374_422204999475172_9039387325224382577_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=e280be&_nc_ohc=ywg34ZgxmvkQ7kNvgFTgDIS&_nc_ht=scontent.fein1-1.fna&oh=00_AYDrz84zpw_D3VAbHiDGOj6nkyWTbGvgqhCiJccjuI7mRQ&oe=668454BA). We use [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) library to train and evaluate session-based recommendation models on the [RecSys Challenge 2024 dataset](https://recsys.eb.dk/dataset/).

## Reproducibility Instructions

### Installation

To install required libraries simply run the command:
`pip install git+https://github.com/NVIDIA/dllogger.git`
`pip install --extra-index-url=https://pypi.nvidia.com -r requirements.txt`

> Alternatively, you can use the provided `environment.yml` file to create a conda environment.

### Data Preparation

First, you can download the data by running script below:
`sh ./scripts/download_data.sh`

Then, you can preprocess the data by running the script below:

```bash
PYTHONPATH=$(pwd) python scripts/preprocess_data.py
PYTHONPATH=$(pwd) python scripts/chunk_data.py
```

The scripts require pointing to the correct data path in the constants at the top of the script.

### Training and Evaluation

To train and evaluate the model, we use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. You can find configuration files in the `configs` directory.

To train the model you can either run XLNet or GPT-2 based model by running the following command:

```bash
PYTHONPATH=$(pwd) python scripts/xlnet_mlm.py
PYTHONPATH=$(pwd) python scripts/gpt2_ntp.py
```

Our training pipelines support direct logging to the [Weights & Biases](https://wandb.ai/) platform. To make sure it works, follow the configuration steps in the [official documentation](https://docs.wandb.ai/quickstart).
