# CS598-Final-Project

This repository contains the code to reproduce the results of the paper A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models by Stefan Hegselmann, Shannon Zejiang Shen, Florian Gierse, Monica Agrawal, David Sontag, and Xiaoyi Jiang.

## Overview

* [gpt-4](https://github.com/kavya-kar/CS598-Final-Project/tree/main/gpt-4): All code related to the GPT-4 experiments.
* [labeling](https://github.com/kavya-kar/CS598-Final-Project/tree/main/labeling): Scripts to analyse and work with labeling data.
* [note](https://github.com/kavya-kar/CS598-Final-Project/tree/main/note): Initial dataset used before processing (modified from original 331793 entries to last 7500 entries for processing constraints).
* [notebooks](https://github.com/kavya-kar/CS598-Final-Project/tree/main/notebooks): Jupyter notebook for different experiments, helper tasks, and analyses.
* [preprocess](https://github.com/kavya-kar/CS598-Final-Project/tree/main/preprocess): Preprocessing pipeline as presented in the paper.
* [scripts](https://github.com/kavya-kar/CS598-Final-Project/tree/main/scripts): Scripts for parameter tuning of LED model.
* [summarization](https://github.com/kavya-kar/CS598-Final-Project/tree/main/summarization): All code related to the summarization experiments with LED model.

## Setting Correct Paths

We assume the root path to be `/root` in this README and for the code. Hence, we assume the repository is cloned to `/root/CS598-Final-Project`. Please adapt the paths according to your local setup.

## Preparing the Environment

We used `conda` to create the necessary virtual environments. For the `ps_llms` environment, we used python 3.9.18:

```
conda create -n ps_llms python==3.9.18
conda activate ps_llms
```

Next, install the necessary requirements:

```
pip install torch torchvision torchaudio
pip install transformers=4.25.1 bitsandbytes sentencepiece accelerate datasets=2.16.0 peft trl py7zr scipy wandb evaluate rouge-score sacremoses sacrebleu seqeval bert_score swifter bioc medcat plotly nervaluate nbformat kaleido
pip install -U spacy
python -m spacy download en_core_web_sm
```