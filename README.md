# 3D-Multi-resolution-Deep-Learning-Model-for-Diagnosis-of-Multiple-Pathological-Types-on-Pulmonary-Nodules

This repository contains the Keras implementation using Tensorflow as backend.

Requirements:

Python 3.7.6 /n
numpy 1.18.5/n
keras 2.3.1/n
tensorflow 1.15.0/n
matplotlib 3.1.3/n
sklearn 0.22.1/n
scipy 1.4.1/n
pandas 1.0.1/n
opencv-python 4.1.1/n

Usage：

1. Run the code：python transfor2npy.py to extract pulmonary nodules from lung CT.
2. Run the code：python augmention.py to realize offline data expansion.
3. If you want to train the model, download the dataset then run the code：python Train_Test_Mutl.py Note that the parameters and paths     should be set beforehand.
4. Once the training is complete, you can modify the Train_Test_Mutl.py file and then run the Train_Test_Mutl.py to test your model.

