# Optimization Benchmark

## Adaptive Gradient Methods Converge Faster with Over-Parameterization (and you can do a line-search) [[Paper]](https://arxiv.org/abs/2006.06835)

Our `AMSGrad Armijo SLS` and `AdaGrad Armijo SLS`  consistently achieve best generalization results.
![](results/results_sls.png)
## Networks Implemented
* linear

* logistic

* mlp

* wrn

* mlp_dropout

* resnet34

* resnet34_100

* resnet34_200

* resnet34_nobn

* resnet34_100_nobn

* resnet18

* resnet50

* resnet101

* resnet152

* mxresnet50

* wide_resnet101

* densenet121

* densenet121_100

* densenet121_nobn

* densenet121_100_nobn

* matrix_fac_1

* matrix_fac_4

* matrix_fac_10

* linear_fac

## Dataloaders Implemented
* tiny_imagenet

* imagenette2-160

* imagewoof2-160

* mnist

* cifar10

* cifar100

* synthetic

* matrix_fac

* mushrooms

* w8a

* rcv1

* ijcnn

* a1a

* a2a

* mushrooms_convex

* w8a_convex

* rcv1_convex

* ijcnn_convex

* a1a_convex

* a2a_convex

## Optimizers Implemented
* adaptive_second

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

## Install requirements
`pip install -r requirements.txt` 


Install the Haven library for managing the experiments with 

```
pip install --upgrade git+https://github.com/haven-ai/haven
```

* torch>=1.4.0

* torchvision>=0.5.0

* pylidc>=0.2.1

* matplotlib>=3.1.2

* scikit-image>=0.14.2

* backpack >= 0.0

* ipdb >= 0.0

* tqdm >= 0.0

* pandas >= 0.0

* scikit-learn >= 0.0

* backpack-for-pytorch >= 0.0


## Usage

**Setup config file**

**To run the experiments :**

```
python trainval.py -e <expconfig> -r "0" -d <datadir> -sb <savedir_base> -nw "0" -j "1"
```

where `<expconfig>` is the name definition of the experiment experiment configuration, `<datadir>` is where the data is saved, and `<savedir_base>` is where the results will be saved.

**To view the results :**

```
python trainval.py -e <expconfig> -v 1 -d <datadir> -sb <savedir_base>
```

where `<expconfig>` is the name definition of the experiment experiment configuration, `<datadir>` is where the data is saved, and `<savedir_base>` is where the results will be saved.



