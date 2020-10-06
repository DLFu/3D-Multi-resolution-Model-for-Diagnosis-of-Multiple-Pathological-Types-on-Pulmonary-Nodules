# 3D Multi resolution Deep Learning Model for Diagnosis of Multiple Pathological Types on Pulmonary Nodules

This repository contains the Keras implementation using Tensorflow as backend.

## Requirements:

Python 3.7.6

numpy 1.18.5

keras 2.3.1

tensorflow 1.15.0

matplotlib 3.1.3

sklearn 0.22.1

scipy 1.4.1

pandas 1.0.1

opencv-python 4.1.1



## Usage：

1. Run the code：python transfor2npy.py to extract pulmonary nodules from lung CT.
2. Run the code：python augmention.py to realize offline data expansion.
3. If you want to train the model, download the dataset then run the code：python Train_Test_Mutl.py Note that the parameters and paths     should be set beforehand.
4. Once the training is complete, you can modify the Train_Test_Mutl.py file and then run the Train_Test_Mutl.py to test your model.

# LICENSE

Code can only be used for ACADEMIC PURPOSES. NO COMERCIAL USE is allowed. Copyright © School of Mechanical,Electrical & Information Engineering, Shandong University. All rights reserved.
