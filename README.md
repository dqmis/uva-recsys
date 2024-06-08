# Contributors
<p align="left">
  <img src="https://contributors-img.web.app/image?repo=ebanalyse/ebnerd-benchmark" width = 50/>
</p>

# Introduction
Hello there 👋🏽

We recommend to check the repository frequently, as we are updating and documenting it along the way!

## EBNeRD 
Ekstra Bladet Recommender System repository, created for the RecSys'24 Challenge. 

# Getting Started
We recommend [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html#conda-environment) for environment management, and [VS Code](https://code.visualstudio.com/) for development. To install the necessart packages and run the example notebook:

```
# 1. Create and activate a new conda environment
conda create -n <environment_name> python=3.11
conda activate <environment_name>

# 2. Clone this repo within VSCode or using command line:
git clone https://github.com/ebanalyse/ebnerd-benchmark.git

# 3. Install the core ebrec package to the enviroment:
pip install .
```

We have experienced issues installing *tensorflow* for M1 Macbooks (```sys_platform == 'darwin'```) when using conda. To avoid this, we suggest to use venv if running on macbooks.
```
python3 -m venv venv
source  venv/bin/activate
```

## Running GPU
```
tensorflow-gpu; sys_platform == 'linux'
tensorflow-macos; sys_platform == 'darwin'
```

# Algorithms
To get started quickly, we have implemented a couple of News Recommender Systems, specifically, 
[Neural Recommendation with Long- and Short-term User Representations](https://aclanthology.org/P19-1033/) (LSTUR),
[Neural Recommendation with Personalized Attention](https://arxiv.org/abs/1907.05559) (NPA),
[Neural Recommendation with Attentive Multi-View Learning](https://arxiv.org/abs/1907.05576) (NAML), and
[Neural Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/) (NRMS). 
The source code originates from the brilliant RS repository, [recommenders](https://github.com/recommenders-team/recommenders). We have simply stripped it of all non-model-related code.


# Notebooks
To help you get started, we have created a few notebooks. These are somewhat simple and designed to get you started. We do plan to have more at a later stage, such as reproducible model trainings.
The notebooks were made on macOS, and you might need to perform small modifications to have them running on your system.

## Model training
We have created a [notebook](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/nrms_ebnerd.ipynb) where we train NRMS on EB-NeRD - this is a very simple version using the demo dataset.

## Data manipulation and enrichment
In the [dataset_ebnerd](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/dataset_ebnerd.ipynb) demo, we show how one can join histories and create binary labels.


## Launching Jupyter session in Snellius
1. Launch an ssh session with port forwarding using your credentials: ` ssh -L 8888:localhost:8888  -i <path_to_ssh_key>  <username>@snellius.surf.nl`
2. Create new interactive session within compute node: `srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=00:10:00 --pty bash -i`
3. Activate conda env where jupyter lab / notebook is installed and then launch new jupyter session without browser: `jupyter lab --no-browser`
4. While keeping the previous ssh session open, launch a new one with the command in step 1.
5. Within Snellius shh session, connect to compute node via ssh port forwarding: `ssh <mode_name> -L 8888:localhost:8888`

## Dataset insights

* The target value (article_ids_clicked) is a list of articles clicked during the impression. Why don't transform it into multiple rows with a single article_id_cliked? Also 99.5% of lists have a single article in it (no need to do multiple articles classification).
* Small and demo datasets are subsets of large dataset (no need to combine them)
* Test set doesn't contain article_id - it means we have to rely on other articles belonging to the same session from train set to. 6% of test set impressions belong to one of the sessions present in the train dataset.
* Big majority of sessions (more than 50%) have a length of 1. These were discarded in the original paper.
* User history spans across 20 days, while impressions span across 6 days (from train data; test data is the same). As soon as user history timestamps end, the impression timestamp begin - we cannot assign recent context from user history.
* Most of the columns containing NaNs can be discarded - no need to spend to much time trying to fill them.