#%%
import torch
import matplotlib.pyplot as plt 
import numpy as np
from surroundmodulation.utils.misc import pickleread, picklesave
from tqdm import tqdm
from scipy.optimize import curve_fit
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2 


def GaborFilterFunction(xy, coeff, pos_x, pos_y, sigma_x, aspect_ratio, sf, phase, theta):
    X, Y = xy
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    sigma_y = sigma_x/aspect_ratio
    pos = np.array([pos_x, pos_y])
    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    pos_x, pos_y = np.matmul(rotation_matrix, pos)
    A = np.exp(
        -0.5
        * (
            ((X_rot - pos_x) ** 2 / sigma_x ** 2)
            + ((Y_rot - pos_y) ** 2 / sigma_y ** 2)
        )
    )
    B = np.cos(2 * np.pi * (X_rot + pos_x) * sf + phase)
    return coeff * A * B


def CenteredGaborFilterFunction(xy, coeff, sigma_x, sigma_y, sf, phase):
    theta=0
    pos_x=0
    pos_y=0
    X, Y = xy
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    pos = np.array([pos_x, pos_y])
    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    pos_x, pos_y = np.matmul(rotation_matrix, pos)
    A = np.exp(
        -0.5
        * (
            ((X_rot + pos_x) ** 2 / sigma_x ** 2)
            + ((Y_rot + pos_y) ** 2 / sigma_y ** 2)
        )
    )
    B = np.cos(2 * np.pi * (X_rot + pos_x) * sf + phase)
    return coeff * A * B


def gaussian2d(xy, pos_x, pos_y, sigma_x, sigma_y):
    theta=0
    X, Y = xy
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    pos = np.array([pos_x, pos_y])
    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    pos_x, pos_y = np.matmul(rotation_matrix, pos)
    A = np.exp(
        -0.5
        * (
            ((X_rot - pos_x) ** 2 / sigma_x ** 2)
            + ((Y_rot - pos_y) ** 2 / sigma_y ** 2)
        )
    )
    return A

import math
def distance_between_angles(angle1, angle2):
    return math.atan2(math.sin(angle2 - angle1), math.cos(angle2 - angle1))

#%% further select cells

d = pickleread('/project/experiment_data/convnext/data_final_1_opt.pickle')
d1 = pickleread('/project/experiment_data/convnext/data_final_2_opt.pickle')
d.update(d1)
print(len(d.keys()))

d_classical = pickleread('/project/experiment_data/convnext/final_classical_data.pickle')
print(len(d_classical.keys()))

idxs = list(d_classical.keys())

#%% fit here
results = {}

bounds=[
        [0, -np.inf, -np.inf, 0, 0, 1, -np.inf, -np.inf], 
        [np.inf, np.inf, np.inf,np.inf , 1, np.inf, np.inf, np.inf]
    ]

for i in idxs:
    converged=False
    # %% # do the fitting 
    x = np.linspace(-2.35/2, 2.35/2, 93)
    X, Y = np.meshgrid(x, x)
    x_data = X.ravel()
    y_data = Y.ravel()
    z_data = d[i]['mei'].squeeze().ravel()
    z_data_std = z_data.std()
    z_data_norm = z_data/z_data_std
    error_best = 1000000
    for j in tqdm(range(20)):
        sigma_x = 0.15
        # sigma_y = 0.2 
        aspect_ratio=1
        phase = float(np.random.rand(1)*2*np.pi)
        ori = float(np.random.rand(1)*2*np.pi)
        sf = 2.
        p0 = [5, 0, 0, sigma_x, aspect_ratio, sf, phase, ori]
        try:
            params, _ = curve_fit(
                GaborFilterFunction, 
                (x_data, y_data), 
                z_data_norm, p0=p0,  
                bounds=bounds)
            fitted_gabor_norm = GaborFilterFunction((x_data, y_data), *params)
            mse = np.mean((z_data_norm - fitted_gabor_norm)**2)
            fitted_gabor = fitted_gabor_norm*z_data_std
            if mse < error_best:
                error_best = mse
                fitted_best = fitted_gabor.reshape(93,93)
                best_params= params
                converged=True
        except Exception as e:
            print(e)
    if converged:
        results[i] = {
            'mse':error_best, 
            'params':best_params,
        }
        # plt.imshow(np.concatenate([GaborFilterFunction((x_data, y_data), *best_params).reshape(93,93)*z_data_std, d[i]['mei'].reshape(93,93)], axis=-1), vmin=-.5, vmax=.5)
        # plt.title(error_best)
        # plt.show()

#%%
picklesave('results_gb_fitting.pickle', results)
#%% select 
results = pickleread('results_gb_fitting.pickle')
#%%
good_idxs = []
x = np.linspace(-2.35/2, 2.35/2, 93)
X, Y = np.meshgrid(x, x)
x_data = X.ravel()
y_data = Y.ravel()
for i in tqdm(idxs): 
    try:
        params = results[i]['params']
        sf = params[5]
        nx = 4*params[3]*sf
        mse = results[i]['mse']
        if sf >1 + 1e-5 and nx > .8 and mse < .3:
            good_idxs.append(i)
    except: 
        pass
print(len(good_idxs))

#%% modified pipeline

image_types = ['exc_full_surr', 'inh_full_surr', 'exc_only_surr', 'inh_only_surr']


from copy import deepcopy
from surroundmodulation.analyses import translate_image, rotate_image, find_extremes_of_mask

def find_first_true(array):
    for i in range(len(array)):
        if array[i]==True:
            return i

def find_last_true(array):
    for i in range(len(array)):
        if array[-i]==True:
            return -i

def find_first_and_last_true(array):
    first = find_first_true(array)
    last = find_last_true(array)
    return first, last

def find_extremes_of_mask2(mask):
    l0, l1 = mask.shape
    mask = mask>0.5
    dim_0 = mask.max(axis=1)
    first_0, last_0 = find_first_and_last_true(dim_0)
    dim_1 = mask.max(axis=0)
    first_1, last_1 = find_first_and_last_true(dim_1)
    return first_0, l0+last_0, first_1, l1+last_1 

# coeff px py sx gamma sf phase ori

bounds=np.array([
        [-np.inf, -np.inf, -np.inf, 0,      0,   1,     -np.inf, -np.inf], 
        [np.inf, np.inf,  np.inf, np.inf , 1.5, np.inf, np.inf, np.inf]
    ])

final_res = {}
results_gauss_gabor_fits = {}
for image_type in image_types:
    for ii, i in enumerate(list(good_idxs)):
        print('neuron n =', ii, image_type)
        np.random.seed=42
    
        results_gauss_gabor_fits[i] = {}
        mask = d[i]['mask_mei']
        params = results[i]['params']
        px = -params[1] # fix
        py = -params[2] # fix
        ori = params[-1]

        gb = GaborFilterFunction((x_data, y_data), *params).reshape(93,93)
        # grating = GaborFilterFunction((x_data, y_data), *params_).reshape(93,93)

        mei = d[i]['mei']
        image = d[i][image_type]
        angle = np.rad2deg(ori)
        
        
        tpx = ((px +(2.35/2))/(2.35/2) )*93/2 - 93/2 
        tpy = ((py +(2.35/2))/(2.35/2) )*93/2 - 93/2 

        translated_mei = translate_image(mei, tpx, tpy)
        translated_mask = translate_image(mask, tpx, tpy)
        translated_gb = translate_image(gb, tpx, tpy)
        translated_image = translate_image(image, tpx, tpy)
        

        mei = rotate_image(translated_mei, angle)
        gb = rotate_image(translated_gb, angle)
        mask = rotate_image(translated_mask, angle) > 0.5
        image = rotate_image(translated_image, angle) 

        x = np.linspace(-2.35/2, 2.35/2, 93)
        X, Y = np.meshgrid(x, x)
        x_data = X.ravel()
        y_data = Y.ravel()

        first_0, last_0 , first_1, last_1 = find_extremes_of_mask2(mask)
        # print(find_extremes_of_mask2(mask))
        # plt.imshow(mask)
        # plt.vlines([first_1, last_1], 0, 93)
        # plt.hlines([first_0, last_0], 0, 93)
        # plt.show()

        # convert in degrees
        top = ((first_0 -(93/2))/(93/2) )*2.35/2 
        bottom = ((last_0 - (93/2))/(93/2) )*2.35/2 
        left = ((first_1 - (93/2))/(93/2) )*2.35/2 
        right = ((last_1 - (93/2))/(93/2) )*2.35/2  

        pos_dict = {
            'top':[0,top],
            'bottom':[0,bottom],
            'left':[left,0],
            'right':[right, 0],

            # 'top1':[0,top - 0.45],
            # 'bottom1':[0,bottom + 0.45],
            # 'left1':[left-0.45,0],
            # 'right1':[right + 0.45, 0],

            'top2':[0,top - 0.45],
            'bottom2':[0,bottom + 0.45],
            'left2':[left-0.45,0],
            'right2':[right + 0.45, 0],
        }
        

        for name, pos in pos_dict.items():
            gauss = np.clip(gaussian2d((x_data,y_data), *pos, 0.3, 0.3).reshape(93,93)-0.3, 0, None)
            
            # plt.imshow(gauss + mask)
            # plt.title(name)
            # plt.vlines([first_1, last_1], 0, 93)
            # plt.hlines([first_0, last_0], 0, 93)
            # plt.show()
            masked_img = gauss*image 

            converged=False
            # %% # do the fitting 
            x = np.linspace(-2.35/2, 2.35/2, 93)
            X, Y = np.meshgrid(x, x)
            x_data = X.ravel()
            y_data = Y.ravel()
            z_data = masked_img.squeeze().ravel()
            z_data_std = z_data.std()
            z_data_norm = z_data/z_data_std

            error_best = 1
            for j in range(30):
                sigma_x = 0.2
                aspect_ratio = 1
                phase = float(np.random.rand(1)*2*np.pi)
                ori = float(np.random.rand(1)*2*np.pi)
                sf = 2.5
                p0 = [1, *pos, sigma_x, aspect_ratio, sf, phase, ori]
                # plt.imshow(GaborFilterFunction((x_data, y_data), *p0).reshape(93,93))
                # plt.show()
                try:
                    params, _ = curve_fit(
                        GaborFilterFunction, 
                        (x_data, y_data), 
                        z_data_norm, 
                        p0=p0, 
                        # maxfev=1000, 
                        # bounds=bounds,
                        )
                    fitted_gabor_norm = GaborFilterFunction((x_data, y_data), *params)
                    # print(params)
                    # plt.imshow(np.concatenate([fitted_gabor_norm.reshape(93,93), z_data_norm.reshape(93,93)], -1))
                    # plt.show()
                    fitted_gabor = fitted_gabor_norm*z_data_std
        
                    mse = np.mean((z_data_norm - fitted_gabor_norm)**2)
                    if mse < error_best:
                        error_best = mse
                        fitted_best = fitted_gabor.reshape(93,93)
                        best_params= params
                        converged=True
                except Exception as e:
                    print(e)

            results_gauss_gabor_fits[i]['original'] = image
            if converged:
                results_gauss_gabor_fits[i][name] = {
                    'image': fitted_best,
                    'params': best_params,
                    'normalized_error': error_best,
                    'image_to_fit': z_data.reshape(93,93),
                }
                # print(results_gauss_gabor_fits[i][name]['params'])
                # plt.imshow(results_gauss_gabor_fits[i][name]['image'])
                # plt.colorbar()
                # plt.show()
                # plt.imshow(results_gauss_gabor_fits[i][name]['image_to_fit'])
                # plt.colorbar()
                # plt.show()
                # plt.imshow(results_gauss_gabor_fits[i]['original'])
                # plt.colorbar()
                # plt.show()
                

        # plt.subplots(figsize=(30,30))
        # plt.imshow(np.concatenate([
        #         image, 
        #         results_gauss_gabor_fits[i]['center']['image'], 
        #         results_gauss_gabor_fits[i]['top']['image'],
        #         results_gauss_gabor_fits[i]['top2']['image'],
        #         results_gauss_gabor_fits[i]['bottom']['image'],
        #         results_gauss_gabor_fits[i]['bottom2']['image'],
        #         results_gauss_gabor_fits[i]['right']['image'],
        #         results_gauss_gabor_fits[i]['right2']['image'],
        #         results_gauss_gabor_fits[i]['left']['image'],
        #         results_gauss_gabor_fits[i]['left2']['image'],
        #         ],
        #         -1
        #     ),
        #     vmin = -.5,
        #     vmax=.5)
        # plt.show()
    
    final_res[image_type]=deepcopy(results_gauss_gabor_fits)
    picklesave('final_res_no_bounds.pickle',  final_res)

# %%