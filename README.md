# Semi-supervised Learning strategy using Deep Network convolutional auto encoders (CAE)
Semi-supervised Learning method for image classification (1k classes). Structure: CNN with residual connections stacked on top of the encoder module of a convolutional auto-encoder (Pytorch). The encoder network weights are trained using unlabeled images after which they are frozen. A CNN with residual connections is then stacked on top and trained on labeled data for classification.

Run [CAE_train.sbatch](../CAE_ResNet/CAE_train.sbatch) file to train and evaluate this Semi-supervised learning strategy.

## Mean Teacher method
The mean teacher code is originated from [Mean Teacher repository](https://github.com/CuriousAI/mean-teacher/tree/master/pytorch). The original one comes with resnet152 and cifar_shakeshake26 model architectures. Here, we added a ResNet18 to the architectures.py but feel free to use any model that is compatible with your purposes. We also modified the main.py script so that it works with the lastest pytorch version.

Before runnig the code, put the training dataset and validation set under the folder: [images/ilsvrc2012](../mean_teacher/data-local/images/ilsvrc2012/) with the folder name 'train' and 'val'. Put your records of labeled data in a separate folder under /Semi-supervised_Learning_DL/mean_teacher/data-local/labels in a txt file with a desired format.

The fine-tune.sbatch file provides a suggestion of hyperparamters to use, but it is not necessary the most optimized version. The sbatch file can be used to submit jobs directly to NYU HPC.

In order to see how to do the training, run python [main.py](../mean_teacher/main.py) --help to get more details.

[Manuscript overaleaf link](https://www.overleaf.com/read/drqnsxrkkxxb}

