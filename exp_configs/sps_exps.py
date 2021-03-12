from haven import haven_utils as hu
import itertools
# RUNS = [0, 1]
# RUNS = [0,1,2,3,4]
RUNS = [0]


def get_benchmark(benchmarks, opt_list):
    if not isinstance(benchmarks, list):
        benchmarks = [benchmarks]

    exp_list = []
    for benchmark in benchmarks:
        if benchmark == 'syn':
            exp_dict =  {"dataset": ["synthetic"],
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
            exp_dict =  {"dataset": ["mushrooms", "ijcnn", "rcv1"],
                    "model_base": ["logistic"],
                    "loss_func": ['softmax_loss'],
                    "score_func": ["softmax_accuracy"],
                    "opt": opt_list,
                    "batch_size": [100],
                    "max_epoch": [100],
                    "runs": RUNS}

        elif benchmark == 'mf':
            exp_dict =  {"dataset": ["matrix_fac"],
                    "model_base": ["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                    "loss_func": ["squared_loss"],
                    "opt": opt_list,
                    "score_func": ["mse"],
                    "batch_size": [100],
                    "max_epoch": [100],
                    "runs": RUNS}

        elif benchmark == 'mnist':
            exp_dict =  {"dataset": ["mnist"],
                    "model_base": ["mlp"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "score_func": ["softmax_accuracy"],
                    "batch_size": [128],
                    "max_epoch": [100],
                    "runs": RUNS}

        elif benchmark == 'cifar10':
            exp_dict =  {"dataset": ["cifar10"],
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
            exp_dict =  {"dataset": ["cifar100"],
                    "model_base": [
                # "densenet121_100",
                "resnet34_100"
            ],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}
        exp_list += hu.cartesian_exp_group(exp_dict)

    return  exp_list

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
syn_dataset = ['syn']
deep_dataset = ['cifar10', 'mnist',  'cifar100']
# Fixed Momentum
same_momentum = []
for c in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    same_momentum += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':c}]
    same_momentum += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':0}]

against_sgd_momentum = [{'name': "sps", 'c': .2,  'adapt_flag': 'mom1', 'momentum':0}, 
                        {'name': "sgd", 'lr': 1e-3,  'momentum':0.9}]


# beta_against_c = []
# # beta in (0,1) and c > (1 - beta)/2
# c_func = lambda beta: (1.-beta)/2.
# for beta in [.2,.4,.6,.8]:
#     c = c_func(beta)
#     beta_against_c += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':beta}]

# Fixed C
beta_against_c2 = []
# beta in (0,1) and c > 1/2(1 - beta)
c_func = lambda beta: 1./(2*(1.-beta))
for beta in [0.,.2,.4,.6,.8]:
    c = c_func(beta)
    beta_against_c2 += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':beta}]
beta_against_c2 += [{'name': "sps", 'c': .2,  'adapt_flag': 'mom1', 'momentum':0}]

# C=.2
fixed_c = [{'name': "sps", 'c': .2,  'adapt_flag': 'mom1', 'momentum':0}]
for mom in [.2, .4,.6,.8]:
    fixed_c += [{'name': "sps", 'c': .2,  'adapt_flag': 'mom1', 'momentum':mom}]

# old_list
old_list = [{'name': "sps", 'c': .2,  'adapt_flag': 'mom1', 'momentum':0}]
# old_list += [{'name': "sps", 'c': 5,  'adapt_flag': 'mom1', 'momentum':0.9}]
old_list += [{'name': "sps", 'c': 1,  'adapt_flag': 'mom1', 'momentum':0.9}]
    # if beta != 0:
    #     beta_against_c2 += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':0}]
# beta_against_c2 = []
#     # beta in (0,1) and c > 1/2(1 - beta)
#     c = lambda beta: 1./(2*(1.-beta))
#     for c in [.2,.4,.6,.8]:
#         beta_against_c2 += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':c}]

# sps_list_mar_5 = [
           

#             ]
# c_list = [.2, .5, 1.0]
# for c in c_list:
#     # sps_list += [{'name': "sps", 'c': c,  'adapt_flag': 'smooth_iter'}]
#     sps_list += [{'name': "sps", 'c': c,  'adapt_flag': 'mom1', 'momentum':0.1}]
opt_list = beta_against_c2 + against_sgd_momentum + same_momentum + sps_list_mom_gamma
opt_list = beta_against_c2
opt_list = fixed_c + against_sgd_momentum
opt_list = old_list
# EXP_GROUPS['sps_syn'] = hu.cartesian_exp_group(get_benchmark(benchmark='syn', opt_list=opt_list))
# EXP_GROUPS['sps_mnist'] = hu.cartesian_exp_group(get_benchmark(benchmark='mnist', opt_list=opt_list))
# EXP_GROUPS['sps_cifar10'] = hu.cartesian_exp_group(get_benchmark(benchmark='cifar10', opt_list=opt_list))
# EXP_GROUPS['sps_cifar100'] = hu.cartesian_exp_group(get_benchmark(benchmark='cifar100', opt_list=opt_list))

EXP_GROUPS['fixed_c'] = get_benchmark(benchmarks=deep_dataset, opt_list=fixed_c)
EXP_GROUPS['beta_against_c2'] = get_benchmark(benchmarks=deep_dataset, opt_list=beta_against_c2)
EXP_GROUPS['old_list'] = get_benchmark(benchmarks=deep_dataset, opt_list=old_list)
EXP_GROUPS['same_momentum'] = get_benchmark(benchmarks=deep_dataset, opt_list=same_momentum)
EXP_GROUPS['against_sgd_momentum'] = get_benchmark(benchmarks=deep_dataset, opt_list=against_sgd_momentum)


for alg in ['alg4', 'alg3']:
    sps_alg = []
    for z in [.5]:
        sps_alg += [
            {'name': "sps", 'c': 0.5,  'adapt_flag': alg,  'z':z},
                    ]
    EXP_GROUPS[f'{alg}_syn'] = get_benchmark(benchmarks=syn_dataset, opt_list=sps_alg)
    EXP_GROUPS[f'{alg}_deep'] = get_benchmark(benchmarks=deep_dataset, opt_list=sps_alg)