# Zoutendijks_Method_Attack
The implementation and experiments are based on Python3.11 and Pytorch. 
### Requirements
- pytorch
- torchvision
- numpy
- matplotlib
- torchattacks https://github.com/Harry24k/adversarial-attacks-pytorch
- robustbench https://github.com/RobustBench/robustbench


Command to reproduce the esults of MNIST in Table 2,3,4:  ```python experiment_MNIST_zoutendijk.py```.<br>
Command to reproduce the results of CIFAR10 in Table 5,6,7:  ```python experiment_Cifar10_zoutendijk.py```.<br>
Command to reproduce the results of ImageNet in Table 8,9,10:  ```python experiment_ImageNet_zoutendijk.py```.<br>
Running results will be stored in ```./result```

Models for MNIST are in the repository. Models for CIFAR and ImageNet will be downloaded automatically by robustbench package in running. MNIST and CIFAR datasets would also be downloaded automatically and the images used for ImageNet(ILSVRC2012) are the first 1000 in validation set. <br>
