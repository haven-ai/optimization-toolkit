from haven import haven_utils as hu
import itertools 

RUNS = [0,1,2,3,4]
def get_benchmark(benchmark, opt_list, batch_size=128):
    if benchmark == 'syn':
        return {"dataset":["synthetic"],
                "model":["logistic"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func":["softmax_accuracy"],
                'margin':
                [ 
                    0.05,
                    #  0.1, 
                    # 0.5,  
                    # 0.01, 
                    ],
                "n_samples": [1000],
                "d": 20,
                "batch_size":[100],
                "max_epoch":[200],
                "runs":RUNS}


    elif benchmark == 'B_C':  
        return {"dataset":["B", "C"],
                "model":["logistic"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func":["softmax_accuracy"],
                "batch_size":[1],
                "max_epoch":[200],
                "runs":RUNS}

    elif benchmark == 'kernels':
        return {"dataset":["mushrooms", "ijcnn", "rcv1"],
                    "model":["logistic"],
                    "loss_func": ['softmax_loss'],
                    "acc_func": ["softmax_accuracy"],
                    "opt":opt_list,
                    "batch_size":[100],
                    "max_epoch":[100],
                    "runs":RUNS}

    elif benchmark == 'mf':
        return {"dataset":["matrix_fac"],
                    "model":["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                    "loss_func": ["squared_loss"],
                    "opt": opt_list,
                    "acc_func":["mse"],
                    "batch_size":[100],
                    "max_epoch":[100],
                    "runs":RUNS}

    elif benchmark == 'mnist':
        return {"dataset":["mnist"],
                    "model":["mlp"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[100],
                    "runs":RUNS}

   
    elif benchmark == 'cifar10':
        return {"dataset":["cifar10"],
                    "model":[
                        # "densenet121",                    
                        "resnet34"
                     ],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[200],
                    "runs":RUNS}

    elif benchmark == 'cifar100':
        return {"dataset":["cifar100"],
                    "model":[
                        # "densenet121_100",
                        "resnet34_100"
                        ],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[200],
                    "runs":RUNS}

    elif benchmark == 'cifar10_nobn':
        return {"dataset":["cifar10"],
                    "model":["resnet34_nobn", "densenet121_nobn"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[200],
                    "runs":RUNS}

    elif benchmark == 'cifar100_nobn':
        return {"dataset":["cifar100"],
                    "model":["resnet34_100_nobn", "densenet121_100_nobn"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[200],
                    "runs":RUNS}

    elif benchmark == 'imagenet200':
        return {"dataset":["tiny_imagenet"],
                    "model":["resnet18"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[200],
                    "runs":RUNS}
    elif benchmark == 'imagenet10':
        return {"dataset":["imagenette2-160", "imagewoof2-160"],
                    "model":["resnet18"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "acc_func":["softmax_accuracy"],
                    "batch_size":[128],
                    "max_epoch":[100],
                    "runs":RUNS}

EXP_GROUPS = {}

def get_opt_dict(c, base_opt, reset_option):
    return {'name':'adaptive_first', 
            'c':c, 
            'gv_option':'per_param',
            'base_opt':base_opt,
            'pp_norm_method':'pp_armijo',
            'init_step_size':100, # setting init step-size to 100. SLS should be robust to this
            "momentum":0.,
            'step_size_method':'sls',
            'reset_option':reset_option}



# ------------------ #
# Convex with interpolation

opt_list = [{'name': 'adam', 'lr':1e-3}]
for c in [0.4, 0.5, 0.6, 0.7, 0.8]:        
    opt_list += [get_opt_dict(c, 'amsgrad', 1)]

for benchmark in ['syn']:
    EXP_GROUPS['nomom_%s' % benchmark] = hu.cartesian_exp_group(get_benchmark(benchmark, opt_list))

# ------------------ #
# IV. Larg-scale nonconvex
opt_list = [{'name': 'adam', 'lr':1e-3}]
for c in [0.1, 0.2, 0.3, 0.4, 0.5]:        
    opt_list += [get_opt_dict(c, 'amsgrad', 1)]

for benchmark in ['cifar100']:
    EXP_GROUPS['nomom_%s' % benchmark] = hu.cartesian_exp_group(get_benchmark(benchmark, opt_list))
             








