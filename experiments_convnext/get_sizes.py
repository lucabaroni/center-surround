#%%
import numpy as np
from surroundmodulation.utils.misc import pickleread

idxs = np.arange(458)
corrs = pickleread('experiment_data/convnext/avg_corr.pkl')
idxs = idxs[corrs>0.75]

d = pickleread('experiment_data/convnext/data_opt.pickle')
d_classical = pickleread('experiment_data/convnext/final_classical_data.pickle')

idxs = d_classical.keys()
#%% estimate size of neurons with size tuning:
def diameter(x1s, y1s, x2s, y2s):
    x1 = np.tile(x1s, (len(x2s), 1))
    y1 = np.tile(y1s, (len(y2s), 1))
    diameters = np.sqrt((x2s[:, None]-x1)**2 + (y2s[:, None]-y1)**2)
#     print(x1s.shape, x2s.shape, diameters.shape)
    return np.max(diameters)

all_masks = [d[idx]['mask_mei'] for idx in idxs]

# compute MEI mask size
ratio = 2.67/all_masks[0].shape[1]
y, x = np.indices(all_masks[0].shape)
threshold = 0.5
threshold_mask = np.array([mask>threshold for mask in all_masks])
threshold_indices = np.array([[y[mask], x[mask]] for mask in threshold_mask])
# mei_diameter = np.array([diameter(*indices1, *indices2) for indices1, indices2 in zip(threshold_indices[:-1], threshold_indices[1:])]) * ratio

mei_diameter = np.array([diameter(*indices1, *indices2) for indices1, indices2 in zip(threshold_indices, threshold_indices)]) * ratio
print(mei_diameter.mean())
# %%
diam = np.array([d_classical[idx]['size_tuning_radius_95'] * 2 for idx in idxs])/2.35*2.67
print(diam.mean())
    
# %%
diam = np.array([d_classical[idx]['size_tuning_radius_95'] * 2 for idx in idxs])
print(diam.mean())
# %%
diam = np.array([d_classical[idx]['size_tuning_top_radius'] * 2 for idx in idxs])
print(diam.mean())

# %%
d_classical[0]['fitted_dot_stim_params']

# %%

def max_sigma(s1, s2, r):
    root = (s1**2 - s2**2)**2 + 4*(r**2)*(s1**2)*(s2**2)
    sm2 = 0.5*(s1**2 + s2**2 + np.sqrt( root ))
    return np.sqrt(sm2)


for idx in idxs: 
    params = d_classical[idx]['fitted_dot_stim_params']
    ms = max_sigma(params[3], params[4], params[5])
    d_classical[idx]['fitted_dot_max_sigma'] = ms*2.67/93

dot_size = np.mean(np.array([d_classical[idx]['fitted_dot_max_sigma'] for idx in idxs])*4)
# %%
dot_size.mean()
# %%

from surroundmodulation.utils.misc import rescale
rescale(.2, -1.7869, 2.1919, 0, 255)
# %%
