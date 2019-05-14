# Semi-supervised_Learning_DL
Semi-supervised Learning method for image classification (1k classes). Structure: CNN with residual connections stacked on top of the encoder module of a convolutional auto-encoder (Pytorch).

## Mean Teacher Method
The mean teacher code is originated from https://github.com/CuriousAI/mean-teacher/tree/master/pytorch. The original one comes with resnet152 and cifar_shakeshake26 model architectures. Here, we modified the main.py script to the lastest pytorch version.

Before runnig the code, put the training dataset and validation set under the folder: /Semi-supervised_Learning_DL/mean_teacher/data-local/images/ilsvrc2012/ with the folder name 'train' and 'val'. Put your records of labeled data in a separate folder under /Semi-supervised_Learning_DL/mean_teacher/data-local/labels in a txt file with a desired format.

The find-tune.sbatch file provides a suggestions of hyperparamters to use, but it is not necessary the most optimized version.
