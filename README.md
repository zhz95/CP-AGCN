# FAIGCN: Frequency Attention Informed Graph Convolutional Network
![Image text](https://github.com/zhz95/hzz/blob/master/FAIGCN.png)

Cerebral Palsy Prediction with Frequency Attention Informed Graph Convolutional Networks

Early diagnosis and intervention are clinically considered the paramount part of treating cerebral palsy (CP), so it is essential to design an efficient and interpretable automatic prediction system for CP. We highlight a significant difference between cerebral palsy patients' frequency of human movement and that of the healthy group, which improves prediction performance. However, the existing deep learning-based methods did not use the frequency information of infants movement for CP prediction. This paper proposes a frequency attention informed graph convolutional network and validates it on two consumer-grade RGB video datasets, namely MINI-RGBD and RVI-38 dataset. Our proposed frequency attention module provides visualization of essential joints considered by the network in CP prediction. We design a frequency-binning method that retains the critical frequency of the human joint position data while filtering the noise. We employ an ablation study to validate the advantage of our method and provide an automatic empirical algorithmic. Our prediction performance achieves state-of-the-art research on both datasets.

For the full MINI-RGBD dataset, please refer to https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html

For the full RVI-38 dataset, please refer to https://github.com/edmondslho/Pose-basedCerebralPalsyPrediction

# Initialization
python >= 3.7

pytorch

# Library requirement and installation
`
pip install requirement.txt
`
`
cd torchlight; python setup.py install; cd ..
`
# Getting started

The command of a quick training with Leave-One-Out Cross-Validation on the MINI-RGBD dataset.
```
python start.py
```

