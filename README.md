# CP-AGCN: A Pytorch-based Attention Informed Graph Convolutional Networks for Cerebral Palsy Classification
![Image text](https://github.com/zhz95/hzz/blob/master/sft.png)

 
The early diagnosis is clinically considered one of the essential parts of cerebral palsy (CP) treatment, so we propose to design a low-cost and interpretable classification system for supporting CP diagnosis. In this work, we implement a Pytorch-based attention-informed graph convolutional network to classify CP patients. This is achieved by integrating the additive attention mechanism into the graph convolutional network. We also propose an optional frequency-binning module to learn the CP movements in the frequency domain while filtering noise. The current version system only requires consumer-grade RGB videos for training to support interactive-time CP diagnosis by providing an interpretable CP classification result. Our flexible system can be further extended to handle other human motion-related disorders (e.g., freezing of gait) and human action recognition tasks.

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

