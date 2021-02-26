from haven import haven_utils as hu
import itertools, copy

EXP_GROUPS = {}

EXP_GROUPS['pascal_all'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': ['pascal'
                                ],
                        # 'dataset_size':{'train':'all', 'val':'all'},
                        'dataset_size':[{'train':50, 'val':10, 'test':10},
                                        {'train':'all', 'val':100, 'test':'all'}],
                        'max_epoch': [20],
                        'optimizer': [
                                {'name':"adasls",'c':0.2}, 
                                 {'name':"sps", 'c':0.2},
                                {'name':"sls",'c':0.2},
                                        
                                        {'name':"adam", "lr":1e-5}, 
                                        
                                      
                                      {'name':"sgd", "lr":5e-5, 'momentum':0.9, 'weight_decay':0.0005}, 
                                      
                                 ], 
                       
                        'model': {'name':'semseg', 'loss':'cross_entropy',
                                  'base':'fcn8_vgg16',
                                   'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS['pascal'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                                
                                {'name':'pascal'} 
                                ],
                        # 'dataset_size':{'train':'all', 'val':'all'},
                        'dataset_size':[{'train':50, 'val':10, 'test':10},
                                        {'train':'all', 'val':100, 'test':'all'}],
                        'max_epoch': [20],
                        'optimizer': [
                                 {'name':"sps", 'c':0.2},
                                {'name':"sls",'c':0.2},
                                        {'name':"adasls",'c':0.2}, 
                                        {'name':"adam", "lr":1e-5}, 
                                        
                                      
                                      {'name':"sgd", "lr":5e-5, 'momentum':0.9, 'weight_decay':0.0005}, 
                                      
                                 ], 
                       
                        'model': {'name':'semseg', 'loss':'cross_entropy',
                                  'base':'fcn8_vgg16',
                                   'n_channels':3, 'n_classes':21}
                        })
