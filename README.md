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


**To run the experiments and get the validation results locally:**

```
python trainval.py -e <expconfig> -r "0" -d <datadir> -sb <savedir_base> -nw "0" -j "0"

<expconfig>             Name definition of the experiment experiment configuration
<datadir>               Path to the saved data directory
<savedir_base>          Path to the saved results directory
```




**To run the experiments and get the validationn results in slurm:**
```
python trainval.py -e <expconfig> -r "0" -d <datadir> -sb <savedir_base> -nw "0" -j "slurm"

<expconfig>             Name definition of the experiment experiment configuration
<datadir>               Path to the saved data directory
<savedir_base>          Path to the saved results directory
```



**To view the results :**

Example
```
python trainval.py -e <expconfig> -v 1 -d <datadir> -sb <savedir_base>

<expconfig>             Name definition of the experiment experiment configuration
<datadir>               Path to the saved data directory
<savedir_base>          Path to the saved results directory
```

## Adding a new benchmark

**Add an optimizer**

1. Define a new optimizer in `src/optimizers/<new_optimizer>.py`.
2. Init the constructor for `opt_name = "<new_optimizer>"` in `src/optimizers/__init__.py`.

For example,
```
elif opt_name == "seg":
        opt = sls_eg.SlsEg(params, n_batches_per_epoch=n_batches_per_epoch)
```

**Add a dataset**

Define a new dataset and its according transformations in `src/datasets/__init__.py` for `dataset_name = "<new_dataset>"`.

For example,
```
   if dataset_name == "mnist":
        view = torchvision.transforms.Lambda(lambda x: x.view(-1).view(784))
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,)),
                                   view
                               ])
                               )
```

**Add a model**

1. Define the matrics, loss functionn, and the accuracy function in the `src/models/classifiers.py`
2. Define the base model in the `get_classifier(clf_name, train_set)` function in `src/models/base_classifiers.py`.


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


* adaptive_first [paper]()

* SGD with Armijo line search [paper](https://msp.org/pjm/1966/16-1/p01.xhtml)

* SGD with Goldstein [paper](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/BF01386306&casa_token=fJPrXJ0xVwIAAAAA:rFFa9IMPl50d2j7xqq3MVrA-L92-O1gdSnlEElXZ7PxnWQYaZQ0LsAWjqjs4TmJb0nHhiNPf1KgVxRhTUw)

* sgd_nesterov [paper](https://ci.nii.ac.jp/naid/10029946121/)

* sgd_polyak [paper](https://d1wqtxts1xzle7.cloudfront.net/51218896/Constrained_Minimization_Methods20170106-3612-16x3v7.pdf?1483706492=&response-content-disposition=inline%3B+filename%3DConstrained_minimization_methods.pdf&Expires=1614407156&Signature=CScszoHzcd4MwWlD1K3Nr1wQ8ukdf~7Y4Z0o9ONncWnUa1UdIhAFNkca7YulYozBOBpPqg1As4A5ChI4r6NGD8~UDb-l3dg5b4s72svQlN1mTgH1kqRanM2~DbGoOJ9Tg4n9drBdZtswe9~mYAyO2LfSlUzAKIWP9qsP8hHxCm9eBVEJOtfU~2erYoVLSzF8-7iN9jO4lnOIv3rJg4zwz470oXK02cOHpwkslr5yQBah4LHV6SpR22JYOQ4Lve6Gl7rdusxX0Txpqbc9UNXOQWoXK~SM-G~iINT1BlQDZepVv4ZMH4~969LV8ORiEAXpyXYZuEPHrWJWULsUd1oEFg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

* adam [paper](https://arxiv.org/pdf/1412.6980.pdf)

* adagrad [paper](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

* ssn [paper](https://arxiv.org/pdf/1910.04920.pdf)

* seg [paper]()

* sgd [paper](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-23/issue-3/Stochastic-Estimation-of-the-Maximum-of-a-Regression-Function/10.1214/aoms/1177729392.full)

* sgd-m [paper]()

* rmsprop [paper](https://arxiv.org/pdf/1308.0850.pdf)

* adabound [paper](https://openreview.net/forum?id=Bkg3g2R9FX)

* amsbound [paper](https://openreview.net/forum?id=Bkg3g2R9FX)

* sps [paper](https://arxiv.org/pdf/2002.10542.pdf)

* lookahead [paper](https://arxiv.org/abs/1907.08610)

* radam [paper](https://arxiv.org/abs/1908.03265)



## Leaderboard
The section is being continually updated with the latest optimizers on standard benchmarks.

### synthetic
![alt text](results/syn.png)

### ijcnn
![alt text](results/ijcnn.png)

### rcv1
![alt text](results/rcv1.png)

### mushrooms
![alt text](results/mushrooms.png)

### w8a
![alt text](results/w8a.png)

### MNIST - MLP
![alt text](results/mnist.png)

### CIFAR10 - ResNet34
![alt text](results/cifar10.png)

### CIFAR100 - ResNet34
![alt text](results/cifar100.png)




