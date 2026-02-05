#%%
import datajoint as dj 
import os 
dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]
import numpy as np
from surroundmodulation.models import SingleCellModel
from tqdm import tqdm
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
from surroundmodulation.utils.misc import pickleread, picklesave
from surroundmodulation.analyses import *

std = 0.05
mean = 0
orientations = np.linspace(0, np.pi, 37)[:-1]
spatial_frequencies = np.linspace(1, 7, 25)
radii = np.linspace(0.0, 2, 41) 
img_res = [93, 93]
size= [2.35,2.35]
gap = 0.2

idxs = np.arange(458)
corrs = pickleread('/project/experiment_data/convnext/avg_corr.pkl')
idxs = idxs[corrs>0.75][:250]

device = f'cuda'

print(device)

all_neuron_model = v1_convnext_ensemble
d = {}

for idx in tqdm(idxs):
    print(idx)
    model = SingleCellModel(v1_convnext_ensemble, idx)
    # optimization exps
    mei, mei_act = create_mei(model, gaussianblur=3., device = device)
    mei_mask,  px_mask, py_mask = create_mask_from_mei(mei, zscore_thresh=1.5)
    exc_full_surr, exc_only_surr, exc_full_surr_act, exc_only_surr_act = create_surround(model, mei, mei_mask, objective='max', gaussianblur=3., device = device, surround_std=.10) # remove surround_std=.10 to get original experiments
    inh_full_surr, inh_only_surr, inh_full_surr_act, inh_only_surr_act = create_surround(model, mei, mei_mask, objective='min', gaussianblur=3., device = device, surround_std=.10) # remove surround_std=.10 to get original experiments

    d[idx] = {
        'mei': mei, 
        'mei_act': mei_act, 
        'mask_mei': mei_mask,
        'center_mask_mei': [px_mask, py_mask],
        'exc_full_surr' : exc_full_surr, 
        'exc_only_surr' : exc_only_surr, 
        'exc_full_surr_act' : exc_full_surr_act, 
        'exc_only_surr_act' : exc_only_surr_act, 
        'inh_full_surr' : inh_full_surr, 
        'inh_only_surr' : inh_only_surr, 
        'inh_full_surr_act' : inh_full_surr_act, 
        'inh_only_surr_act' : inh_only_surr_act, 
        }

    picklesave(f'/project/experiment_data/convnext/data_opt.pickle', d)

