# ReID
Few-shot person re-identification with deep learning.

# Description
The model is implemented from scratch combining various techniques in the referece papers. It combines the triplet loss with semi hard-negative sample mining with the local feature alignment by finding the shortest matching path using dynamic programming as well as the global feature embeddings. I also implemented custom re-ranking algorithms so it's more suitable for rea-world use cases.

# Re-Ranking
Re-ranking is often used in recommender systems and RAGs for retrieve most relevant embeddings.

***TOP-K Re-Ranking***: Evaluates if the relevant items exists within the retrieved top-k items
***TOP-K KNN Re-Ranking***: Use KNN for class identification
***Cluster Re-Ranking***: Use k-means (or average distances to each cluster point) for computing the distance

# Benchmark
Evaluated on Market 1501

|  | mAP@top1-5 | R-1 | R-5 | KNN@top5 | k-means |
| - | - | - | - | - | - |
| ReID (%) | 92.3 |  | 99.6 | 92.7 | 94.1 |

# Dataset
Market 1501 is used to train and evluate the model. You can download the dataset from [here](https://www.kaggle.com/datasets/pengcw1/market-1501/data).

# How to train and evaluate the model
## 1. Prepare the dataset.
For the custom dataset, make sure to follow the format of Market 1501.

## 2. Edit the config file
```rb
./cfg/config.yaml
```

## 3. Train
```rb
python3 train.py
```

## 4. Evaluate
```rb
python3 eval.py
```

# References
[https://arxiv.org/abs/1711.08184](https://arxiv.org/abs/1711.08184)
[https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)
