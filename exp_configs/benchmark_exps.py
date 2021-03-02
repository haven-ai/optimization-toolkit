from haven import haven_utils as hu
RUNS = [0, 1, 2, 3, 4]


def get_benchmark(benchmark, opt_list):
    if benchmark == 'syn':
        return {"dataset": ["synthetic"],
                "model_base": ["logistic"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                'margin': [0.1],
                "n_samples": [1000],
                "d": 20,
                "batch_size": [100],
                "max_epoch": [200],
                "runs": RUNS}

    elif benchmark == 'kernels':
        return {"dataset": ["mushrooms", "ijcnn", "rcv1", "w8a"],
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
                "model_base": ["resnet34"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}

    elif benchmark == 'cifar100':
        return {"dataset": ["cifar100"],
                "model_base": ["resnet34_100"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}


EXP_GROUPS = {}
opt_list = []
sps = [{'name': "sps", 'c': 0.2,  'adapt_flag': 'mom1',
        'epe': 0, 'gamma': 2, 'momentum': 0}]
adaptive_first = [{'name': 'adaptive_first',
                   'c': 0.5,
                   'gv_option': 'per_param',
                   'base_opt': 'adagrad',
                   'pp_norm_method': 'pp_lipschitz',
                   'init_step_size': 1,
                   "momentum": 0.,
                   'step_size_method': 'sls',
                   'beta': 0.99,
                   'gamma': 2,
                   'reset_option': 1,
                   'beta_b': .9,
                   'beta_f': 2.,
                   'line_search_fn': "armijo",
                   'adapt_flag': 'constant'}]
sgd_armijo = [{'name': "sgd_armijo",
               'c': 0.1,
               'reset_option': 1,
               "gamma": 2.0,
               "line_search_fn": "armijo",
               "init_step_size": 1}]
sgd_goldstein = [{'name': "sgd_goldstein",
                  'c': 0.1,
                  'reset_option': 0}]
sgd_nesterov = [{'name': "sgd_nesterov",
                 'gamma': 2.0,
                 "aistats_eta_bound": 10.0}]
sgd_polyak = [{'name': "sgd_polyak",
               'c': 0.1,
               'momentum': 0.6,
               "gamma": 2.0,
               "aistats_eta_bound": 10.0,
               "reset": 0}]
seg = [{'name': "seg"}]
ssn = [{'name': "ssn",
        'init_step_size': 1.0,
        'c': 0.1,
        "lm": 1e-3}]
adam = [{'name': 'adam', 'lr': 1e-3, 'betas': [0.9, 0.99]}]
adagrad = [{'name': 'adagrad', 'lr': 1e-3}]
sgd = [{'name': 'sgd', 'lr': 1e-3}]
sgd_m = [{'name': 'sgd', 'lr': 1e-3, 'momemtum': 0.9}]
rmsprop = [{'name': 'rmsprop', 'lr': 1e-3}]
adabound = [{'name': 'adabound'}]
amsbound = [{'name': 'amsbound'}]
lookahead = [{'name': 'lookahead'}]
radam = [{'name': 'radam'}]
plain_radam = [{'name': 'plain_radam'}]

opt_list = sps + adaptive_first + sgd_armijo + sgd_goldstein + ssn + \
    adam + adagrad + sgd + sgd_m + rmsprop + adabound + \
    amsbound + lookahead + radam + plain_radam
#     + seg + sgd_nesterov + sgd_polyak


EXP_GROUPS['mnist'] = hu.cartesian_exp_group(
    get_benchmark(benchmark='mnist', opt_list=opt_list))
EXP_GROUPS['cifar10'] = hu.cartesian_exp_group(
    get_benchmark(benchmark='cifar10', opt_list=opt_list))
EXP_GROUPS['cifar100'] = hu.cartesian_exp_group(
    get_benchmark(benchmark='cifar100', opt_list=opt_list))
EXP_GROUPS['kernels'] = hu.cartesian_exp_group(
    get_benchmark(benchmark='kernels', opt_list=opt_list))
EXP_GROUPS['syn'] = hu.cartesian_exp_group(
    get_benchmark(benchmark='syn', opt_list=opt_list))
