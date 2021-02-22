# Awesome Optimization Benchmark

The goal of this repository is 
  - to illustrate how different optimizers perform on different datasets using a standard benchmark; and 
  - allow users to add their own datasets and optimizers to have a reliable comparison and inspire new state-of-the-art optimizers for different machine learning problem setups.

## Contents

- Quick Start
- Adding new benchmarks
- Optimizers Implemented
- Leaderboard


## Quick Start (similar to Usage in https://github.com/meetshah1995/pytorch-semseg)


**Install requirements**
`pip install -r requirements.txt` 


**To run the experiments :**

```
python trainval.py -e <expconfig> -r "0" -d <datadir> -sb <savedir_base> -nw "0" -j "1"
```

where `<expconfig>` is the name definition of the experiment experiment configuration, `<datadir>` is where the data is saved, and `<savedir_base>` is where the results will be saved.


```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--dcrf [DCRF]] [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap

```

**To view the results :**

Example
```
python trainval.py -e <expconfig> -v 1 -d <datadir> -sb <savedir_base>
```

where `<expconfig>` is the name definition of the experiment experiment configuration, `<datadir>` is where the data is saved, and `<savedir_base>` is where the results will be saved.

## Adding a new benchmark

**Add an optimizer**

**Add a dataset**

**Add a network**

**Run the new benchmark**

Define the experiment configuration you would like to run. The dataset, models, optimizers, and hyperparameters can all be defined in the experiment configurations.
```
EXP_GROUPS['new_benchmark'] = {"dataset": [<dataset_name>],
                     "model_base": [<network_name>],
                     "opt": [<optimizer_dict>],}
```

Train using the following command
```
python trainval.py -e new_benchmark -v 1 -d ../results -sb ../results
```

### Optimizers Implemented (similar to https://github.com/gjy3035/Awesome-Crowd-Counting - add paper and a one line title of the optimizer which is the paper's name and divide them by year)


* adaptive_first

* sgd_armijo

* sgd_goldstein

* sgd_nesterov

* sgd_polyak

* seg

* ssn

* adam

* adagrad

* sgd

* sgd-m

* rmsprop

* adabound

* amsbound

* sps

* lookahead

* radam

* plain_radam



## Leaderboard
The section is being continually updated with the latest optimizers on standard benchmarks.

Show the training loss and validation accuracy with 5 runs as in https://github.com/IssamLaradji/ada_sls

### MNIST - LeNet

### CIFAR10 - ResNet34

### CIFAR100 - ResNet34




