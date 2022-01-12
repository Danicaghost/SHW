# Image sentiment analysis via active sample refinement and cluster correlation mining

This code is the official implementation of Image sentiment analysis via active sample refinement and cluster correlation mining.

1、Proposed an active sample refinement (ASR) strategy that can adaptively “generate” (not really generating but mining) high-quality images with definite sentiment semantics

2、Created a set of more discriminant but robust features by fully mining the implicit cluster correlation among the heterogeneous SENet features.

# Run

We conducted experiments under

 python 3.6

scikit-learn 0.23.1

modal 0.4.1

Catboost 0.26

Xgboost 1.4.2

# Active Sample Refinement

Run file active.py to refine samples

# Image Feature Extraction

In order to training model, we extract 5 SENet from different layer.They are SENet50,SENet101, SENet152,SENetXT50 and SENetXT101. Therefore, run following file to extract image features:

 

ser2.py, ser101n.py,ser152n.py,serxt50n.py and serxt101n.py

 

# Cluster Correlation Mining

Run NEW_clusterccaCsvd.m to mining cluster correlation

# Image Sentiment Analysis

Run TW5_rc.py to predict, choose top 3 to complete ensemble learning 

# Ensemble Learning

Run My01_01_VOTING_top3_hard.py

 