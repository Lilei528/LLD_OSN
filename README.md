# LLD-OSN

  Authors: Han Zhang, Yazhou Zhang, Wei Wang, and Lixia Ji.

## Overview

  This is the source code for our paper **《LLD-OSN: An effective method for text classification in open-set noisy data》**



#### **What is LLD-OSN?**

  The availability of well-annotated datasets is one of the crucial factors for the remarkable success of deep neural networks, but training data inevitably contain noisy labels in practical applications. Most existing robust methods follow the closed-set assumption, ignoring the impact of out-of-distribution (OOD) noise on generalization performance. This issue reduces the reliability of the systems with real-world consequences. Therefore, we propose Learning Label Distribution in Open Set Noise (LLD-OSN), which classifies the training data into three types and employs tailored strategies for each type, enhancing the robustness of the model. The principle is to use the low-loss strategy and noise classification head to divide samples into clean, out-of-distribution, and ambiguous sets. Subsequently, true label distribution will be learned through the Mahalanobis Distance, Mixup strategy, and flattening techniques. Learning on out-of-distribution samples resolves the issue of overconfidence. Furthermore, we introduce the Co-teaching strategy and soft labels to promote the learning of consistent data features from diverse perspectives. Finally, these components are integrated into a unified optimization objective. Comprehensive experiments on synthetic and real-world datasets validate the effectiveness of LLD-OSN.

## The overall framework

> <img src="/Fig1.png" alt="Figure1" style="zoom: 33%;" />

## The Used Datasets

  Our proposed method is primarily aimed at text classification tasks and is compared with baseline models on four datasets. These datasets include: 

- 20 Newsgroups [link](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)
- AG News and 20 Newsgroups  [link](https://paperswithcode.com/dataset/ag-news)  
-  Yahoo Answers [link](https://paperswithcode.com/dataset/yahoo-answers)
- NoisywikiHow [link](https://github.com/tangminji/noisywikihow)

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.10.1. More details of the environment dependencies required for code execution can be found in the `requirements.txt` file within the repository.

## Experiment

The proposed method is compared with existing noise learning methods:

- BERT [link](https://eva.fing.edu.uy/pluginfile.php/524749/mod_folder/content/0/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.pdf)
- SelfMix  [link](https://arxiv.org/abs/2210.04525)
- Co-teaching  [link](https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)
- PNP [link](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_PNP_Robust_Learning_From_Noisy_Labels_by_Probabilistic_Noise_Prediction_CVPR_2022_paper.html)
- Noise Matrix [link](https://arxiv.org/abs/1903.07507)
- Toward [link](https://www.sciencedirect.com/science/article/pii/S0020025524000732)
- SaFER [link](https://aclanthology.org/2023.acl-industry.38/)
- Rank-aware [link](https://aclanthology.org/2023.tacl-1.45/)
- Neighborhood [link](https://ojs.aaai.org/index.php/AAAI/article/view/26260)
- ReGEN [link](https://aclanthology.org/2023.findings-acl.748/)

## Usage

1. Download the datasets from the link provided above to the `dataset` directory under the root directory. Each line of the data should contain a label and the text content, separated by a tab (\t). The repository includes the correctly formatted datasets. Due to the excessive number of small-noise datasets, they are no longer uploaded. Instead, they can be generated using the script `build_agmix20news_asym.py` or `build_agmix20news_sym.py`.

   --\

      -- dataset

   ​        -- 20newsgroup 

   ​				”train. csv“ and ”test. csv“ are the original datasets, while ”train_noisy_ [A] [B]“. csv is the data injected with A-ratio B-type noise.

   ​        -- wikihow

   ​               ”ood_0.4.csv“ represents out of distribution noise, "id_0.4.csv" represents in distribution noise

   ​        -- yahoo

   ​				”train. csv“ and ”test. csv“ are the original datasets, 

   ​        -- 20newsmixag

   ​              Naming format is the same as 20newsgroup. ag.csv  contains 2000 samples from agnews.

2. Modify the config  file under ./config

   The file contains the following parameters: **database **, **dataset**, **n_classes** (number of classes), **pretrainedmodel** (pretrained model directory), **dict_len** (vocabulary size), **trans_n_head** (number of attention heads in the model), **trans_n_layer** (number of layers in the model), **d_model** (feature vector dimension), and **logging** (log information).

   The configuration information for the dataset is already available.

   --\

      -- config

   ​          xxx.cfg

3. Run the baseline model. You can find the paper and repository address for the baseline model through the link provided above. We have provided simple implementations of some methods. `train_bert.py` is the implementation for the BERT model.

4. Run our proposed method. `train.py` contains the specific implementation of our proposed method. Use the config file and dataset directory as input parameters for execution. Some parameters in this file need to be set.Test 

   **config**: Configuration file directory,
   **log_prefix**: Log directory prefix,
   **log_freq**: Log frequency,
   **threshold**: Mahalanobis distance threshold,
   **net**: Network name,
   **stage1**: Pre-warmup epochs.

5. Test results. During the training process, testing will be performed at regular intervals, and the results will be recorded in the log file.

6. Some code used in the experiments is also provided. For example: `heatmaps.py` generates heatmaps , `distance.py` calculates the distance between samples and class centers, `conf.py` examines confidence scores.

   However, some easily reproducible experimental code (such as ablation studies and case studies) is not provided.

## Experiments

The results compared with the baseline model are as follows:
<div align=center>

![](/Fig2.png)

![Figure3](/Fig3.png)

![Figure3](/Fig4.png)
</div>

The sensitivity of the hyperparameters is as follows:
<div align=center>

![](/Fig5.png)
</div>
The impact of confidence is as follows:
<div align=center>

![](/Fig6.png)
</div>
Visualization:
<div align=center>

![](/Fig7.png)

![](/Fig8.png)
</div>
Computing Resources:
<div align=center>
![](/Fig9.png)</div>
