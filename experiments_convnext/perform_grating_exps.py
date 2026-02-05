#%%
import datajoint as dj 
import os 
dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]
import matplotlib.pyplot as plt

import numpy as np
from surroundmodulation.models import SingleCellModel

from tqdm import tqdm
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
from imagen.image import BoundingBox
from surroundmodulation.utils.misc import pickleread, picklesave
from surroundmodulation.analyses import *

std = 0.05
mean = 0
orientations = np.linspace(0, np.pi, 37)[:-1]
spatial_frequencies = np.linspace(1, 6, 21) #note: change spatial freq
radii = np.linspace(0.0, 2, 41)#this 
img_res = [93, 93]
size= [2.35,2.35]
gap = 0.2

# idxs = np.array(pickleread('/project/experiment_data/convnext/gabor_idx.pickle'))
idxs = np.arange(458)
corrs = pickleread('/project/experiment_data/convnext/avg_corr.pkl')
idxs = idxs[corrs>0.75]


device = f'cuda'

print(device)

all_neuron_model = v1_convnext_ensemble
dot_stim_dict = dot_stimulation(all_neuron_model, idxs)
best_grating_dict = find_preferred_grating_parameters(all_neuron_model, idxs, contrast=0.2, spatial_frequencies=spatial_frequencies)

good_idxs = []
bad_idxs = []
positions = {}
fitted_dot_stim_params = {}
fitted_dot_stim_d = {}

def gaussian2D_with_correlation(xy, A, x0, y0, sigma_x, sigma_y, rho):
    x, y = xy
    a = 1.0 / (2 * (1 - rho**2))
    b = ((x - x0)**2) / (sigma_x**2)
    c = 2 * rho * (x - x0) * (y - y0) / (sigma_x * sigma_y)
    d = ((y - y0)**2) / (sigma_y**2)
    return A * np.exp(-a * (b - c + d))

def gauss_fit(dot_stim_img, img_size=[93,93]): 
    with torch.no_grad():
        data = np.clip(dot_stim_img, a_min=0, a_max=None)
        data = data/data.std()
        x_data = np.arange(0, img_size[1])
        y_data = np.arange(0, img_size[0])
        x, y = np.meshgrid(x_data, y_data)

        # Flatten for fitting
        x_data, y_data = x.ravel(), y.ravel()
        z_data = data.ravel()

        # Initial guess [A, x0, y0, sigma_x, sigma_y, rho]
        initial_guess = [5, 93/2, 93/2, 10, 10, 0]

        # Fit the model
        params, covariance = curve_fit(gaussian2D_with_correlation, (x_data, y_data), z_data, p0=initial_guess)

        # Extract the optimized parameters
        A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt = params
        # print(f"Optimized parameters: A = {A_opt}, x0 = {x0_opt}, y0 = {y0_opt}, sigma_x = {sigma_x_opt}, sigma_y = {sigma_y_opt}, rho = {rho_opt}")

        fitted_data = gaussian2D_with_correlation((x_data, y_data), *params)
        return A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_data.reshape(*img_size)
#%%
good_idxs = []
for idx in idxs:
    try: 
        dot_stim_norm = dot_stim_dict[idx].detach().cpu().numpy()/dot_stim_dict[idx].detach().cpu().numpy().std()
        A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_dot_stim = gauss_fit(dot_stim_norm)
        error = np.mean((fitted_dot_stim - dot_stim_norm)**2)
        if error < 0.2:
            # plt.imshow(np.concatenate([fitted_dot_stim, dot_stim_norm], -1))
            # plt.title(f'{error:.2f}')
            # plt.colorbar()
            # plt.show()
            good_idxs.append(idx)
            positions[idx] = [x0_opt, y0_opt]
            fitted_dot_stim_params[idx] = A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt
            fitted_dot_stim_d[idx] = fitted_dot_stim
        else: 
            # plt.imshow(np.concatenate([fitted_dot_stim, dot_stim_norm], -1))
            # plt.title(f'{error:.2f}')
            # plt.colorbar()
            # plt.show()
            bad_idxs.append(idx)
      
    except: 
        bad_idxs.append(idx)
#%%
#%%
d = {}

for idx in tqdm(good_idxs):
    model = SingleCellModel(v1_convnext_ensemble, idx)
    # optimization exps
    # mei, mei_act = create_mei(model, gaussianblur=3., device = device)
    # mei_mask,  px_mask, py_mask = create_mask_from_mei(mei, zscore_thresh=1.5)
    # print(px_mask, py_mask)
    # print(positions[idx][0], positions[idx][1])
    # plt.imshow(mei_mask)
    # plt.plot(px_mask, py_mask, '*')
    # plt.show()
#     exc_full_surr, exc_only_surr, exc_full_surr_act, exc_only_surr_act = create_surround(model, mei, mei_mask, objective='max', gaussianblur=3., device = device, surround_std=.10) # remove surround_std=.10 to get original experiments
#     inh_full_surr, inh_only_surr, inh_full_surr_act, inh_only_surr_act = create_surround(model, mei, mei_mask, objective='min', gaussianblur=3., device = device, surround_std=.10) # remove surround_std=.10 to get original experiments

    px = positions[idx][0]
    py = positions[idx][1]
    
    top_radius, top_phase, top_grating, top_resp, st_resp, st_gratings = size_tuning_all_phases(
        model, 
        px, 
        py, 
        best_grating_dict[idx]['max_ori'],
        best_grating_dict[idx]['max_sf'], 
        return_all=True, 
        device = device, 
        contrast = 0.2, 
        radii = radii)

    monotonic, radius_95, top_radius_, suppression_index, st_r_95 = assert_size_tuning(radii, np.array(st_resp).max(axis=-1)) #this should be slightly fixed
    
    d[idx] = {
        'dot_stim' : dot_stim_dict[idx],
        'center_dot_stim': [px, py],
        'fitted_dot_stim_params': fitted_dot_stim_params[idx], 
        'fitted_dot_stim': fitted_dot_stim_d[idx],

        'masked_grating_max_ori' : best_grating_dict[idx]['max_ori'],
        'masked_grating_max_sf' : best_grating_dict[idx]['max_sf'],
        'max_phase_max_phase' : best_grating_dict[idx]['max_phase'], 
        'masked_grating_max_stim' : best_grating_dict[idx]['stim_max'],
        'masked_grating_max_resp' :  best_grating_dict[idx]['resp_max'],

        'size_tuning_top_radius' : top_radius,
        'size_tuning_radius_95' : radius_95,
        'size_tuning_top_phase' : top_phase,
        'size_tuning_top_grating' : top_grating,
        'size_tuning_resp' : st_resp,
        'size_tuning_monotonic': monotonic, 
        'size_tuning_suppression_idx': suppression_index, 
        }

    if suppression_index != None and suppression_index>0.05:
        oc_stims, oc_resps = orientation_contrast(
            model, 
            x_pix = px, 
            y_pix= py,
            preferred_ori = best_grating_dict[idx]['max_ori'],
            preferred_sf = best_grating_dict[idx]['max_sf'],
            center_radius=top_radius,
            orientation_diffs = np.linspace(0, np.pi, 37)[:-1], 
            phases = top_phase + np.linspace(0, 2*np.pi, 19)[:-1], 
            gap=gap, 
            contrast = 0.2
        )
        oc_stims_95, oc_resps_95 = orientation_contrast(
            model, 
            x_pix = px, 
            y_pix= py,
            preferred_ori = best_grating_dict[idx]['max_ori'],
            preferred_sf = best_grating_dict[idx]['max_sf'],
            center_radius=radius_95,
            orientation_diffs = np.linspace(0, np.pi, 37)[:-1], 
            phases = top_phase + np.linspace(0, 2*np.pi, 19)[:-1], 
            gap=gap, 
            contrast = 0.2
        )
        d[idx].update({ 
            'oc_resps': oc_resps,
            'oc_resps_95': oc_resps_95,
            # 'oc_stims': oc_stims,
            # 'oc_stims_95': oc_stims_95
            })


    picklesave(f'/project/experiment_data/convnext/final_classical_data.pickle', d)
# %%
