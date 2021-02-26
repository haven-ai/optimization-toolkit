from haven import haven_utils as hu
import itertools
# RUNS = [0, 1]
# RUNS = [0,1,2,3,4]
RUNS = [0]


def get_benchmark(benchmark, opt_list):
    if benchmark == 'syn':
        return {"dataset": ["synthetic"],
                "model_base": ["logistic"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                'margin':
                [
                    0.05,
            0.1,
                    0.5,
                    0.01,
        ],
            "n_samples": [1000],
            "d": 20,
            "batch_size": [100],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'kernels':
        return {"dataset": ["mushrooms", "ijcnn", "rcv1"],
                "model_base": ["logistic"],
                "loss_func": ['softmax_loss'],
                "score_func": ["softmax_accuracy"],
                "opt": opt_list,
                "batch_size": [100],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'mf':
        return {"dataset": ["matrix_fac"],
                "model_base": ["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "score_func": ["mse"],
                "batch_size": [100],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'mnist':
        return {"dataset": ["mnist"],
                "model_base": ["mlp"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'cifar10':
        return {"dataset": ["cifar10"],
                "model_base": [
            # "densenet121",
            "resnet34"
        ],
            "loss_func": ["softmax_loss"],
            "opt": opt_list,
            "score_func": ["softmax_accuracy"],
            "batch_size": [128],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'cifar100':
        return {"dataset": ["cifar100"],
                "model_base": [
            "densenet121_100",
            "resnet34_100"
        ],
            "loss_func": ["softmax_loss"],
            "opt": opt_list,
            "score_func": ["softmax_accuracy"],
            "batch_size": [128],
            "max_epoch": [200],
            "runs": RUNS}


EXP_GROUPS = {}


# 2. SPS
sps_list_gamma = [
            # smooth iter only
            {'name': "sps", 'c': 0.2,  'adapt_flag': 'smooth_iter', 'gamma':2},
            {'name': "sps", 'c': 0.2,  'adapt_flag': 'smooth_iter', 'gamma':1},
]

sps_list_mom = [
            # momentum on top of smooth iter
            {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':.1},
            {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':0.},
            
            {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':.5},
            {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':.9},
            
            ]

sps_list_mom_gamma = [
            # momentum on top of smooth iter
            # {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':0.},
            # {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':.1},
            # {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':0.3},
            # {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':.5},
            # {'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1', 'momentum':.9},
            
            # {'name': "sps", 'c': 0.4,  'adapt_flag': 'mom1', 'momentum':.1},
            # {'name': "sps", 'c': 0.4,  'adapt_flag': 'mom1', 'momentum':0.3},
            # {'name': "sps", 'c': 0.4,  'adapt_flag': 'mom1', 'momentum':.5},
            # {'name': "sps", 'c': 0.4,  'adapt_flag': 'mom1', 'momentum':.9},
            # latest
            # {'name': "sps", 'c': 1.4,  'adapt_flag': 'mom1', 'momentum':.3},
            # {'name': "sps", 'c': 3,  'adapt_flag': 'mom1', 'momentum':.5},
            {'name': "sps", 'c': -.1,  'adapt_flag': 'mom1', 'momentum':-.1},
            {'name': "sps", 'c': 0.5,  'adapt_flag': 'mom1', 'momentum':0.},
            {'name': "sps", 'c': 0.5,  'adapt_flag': 'mom1', 'momentum':.1},
            {'name': "sps", 'c': 0.5,  'adapt_flag': 'mom1', 'momentum':0.3},
            
            # {'name': "sps", 'c': 0.5,  'adapt_flag': 'mom1', 'momentum':.49},
            {'name': "sps", 'c': 0.5,  'adapt_flag': 'mom1', 'momentum':.9},
            
            # {'name': "sps", 'c': 0.5,  'adapt_flag': 'mom1', 'momentum':.5},
            # {'name': "sps", 'c': 0.1,  'adapt_flag': 'mom1', 'momentum':.1},
            # {'name': "sps", 'c': 0.9,  'adapt_flag': 'mom1', 'momentum':.9},

            # {'name': "sps", 'c': 1.,  'adapt_flag': 'mom1', 'momentum':0.},
            # {'name': "sps", 'c': 1.,  'adapt_flag': 'mom1', 'momentum':.1},
            # {'name': "sps", 'c': 1.,  'adapt_flag': 'mom1', 'momentum':0.3},
            # {'name': "sps", 'c': 1.,  'adapt_flag': 'mom1', 'momentum':.5},
            # {'name': "sps", 'c': 1.,  'adapt_flag': 'mom1', 'momentum':.9},

            # {'name': "sps", 'c': 1.4,  'adapt_flag': 'mom1', 'momentum':0.3},

            ]

same_momentum = []
for c in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    same_momentum += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':c}]

# c_list = [.2, .5, 1.0]
# for c in c_list:
#     # sps_list += [{'name': "sps", 'c': c,  'adapt_flag': 'smooth_iter'}]
#     sps_list += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':0.1}]

opt_list = same_momentum + sps_list_mom_gamma

EXP_GROUPS['sps_syn'] = hu.cartesian_exp_group(get_benchmark(benchmark='syn', opt_list=opt_list))
EXP_GROUPS['sps_mnist'] = hu.cartesian_exp_group(get_benchmark(benchmark='mnist', opt_list=opt_list))
EXP_GROUPS['sps_cifar10'] = hu.cartesian_exp_group(get_benchmark(benchmark='cifar10', opt_list=opt_list))
EXP_GROUPS['sps_cifar100'] = hu.cartesian_exp_group(get_benchmark(benchmark='cifar100', opt_list=opt_list))