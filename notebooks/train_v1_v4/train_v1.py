from math import gamma
from nnfabrik.builder import get_model, get_trainer, get_data
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
import datajoint as dj 
import os 


dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]

import numpy as np
for seed in [1, 2,3]:
    for hidden_kern in [7, 9, 11]:
        for input_kern in  [7, 9, 11]:
            for gamma_readout in [0.5, 10, 1, .1,]:

                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
                {'stop_function': 'get_poisson_loss',
                'maximize': False,
                'avg_loss': False,
                'device': 'cuda',
                'max_iter': 200,
                'lr_init': 0.005,
                'lr_decay_steps': 5,
                'patience': 3,
                'verbose': True,
                'adamw': True})

                trainer = get_trainer(trainer_fn, trainer_config)

                dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
                {'dataset': 'CSRF19_V1',
                'neuronal_data_files': ['/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632669014376.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632932714885.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3633364677437.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634055946316.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634142311627.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634658447291.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634744023164.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635178040531.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635949043110.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636034866307.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636552742293.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637161140869.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637248451650.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637333931598.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637760318484.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637851724731.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638367026975.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638456653849.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638885582960.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638373332053.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638541006102.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638802601378.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638973674012.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639060843972.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639406161189.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640011636703.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639664527524.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639492658943.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639749909659.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640095265572.pickle',
                '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle'],
                'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
                'crop': 70,
                'subsample': 1,
                'seed': 1000,
                'time_bins_sum': 12,
                'batch_size': 128})

                dataloaders = get_data(dataset_fn, dataset_config)
                model_fn, model_config = ('nnvision.models.se_core_full_gauss_readout',
                {'pad_input': False,
                'stack': -1,
                'depth_separable': True,
                'input_kern': input_kern,
                'gamma_input': 10,
                'gamma_readout': gamma_readout,
                'hidden_dilation': 1,
                'hidden_kern': hidden_kern,
                'n_se_blocks': 0,
                'hidden_channels': 32})

                model = get_model(model_fn, model_config, dataloaders=dataloaders, seed=0)
                model.cuda().train()
                trainer(model, dataloaders, seed=42)

                torch.save(model.state_dict(), f"mouselike_model_seed={seed}_input_kern={model_config['input_kern']}_hidden_kern={model_config['hidden_kern']}_gamma_readout={model_config['gamma_readout']:.2f}.pt")
                