#%%
from copyreg import pickle
from re import L
from statistics import mean
from surroundmodulation.utils.misc import pickleread
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble, v1_convnext
from nnfabrik.builder import get_model, get_trainer, get_data
import torch
import torch.nn as nn 
import copy
import numpy as np
import matplotlib.pyplot as plt
from surroundmodulation.analyses import create_mask_from_mei
from tqdm import tqdm
import featurevis.ops as ops


from scipy.optimize import curve_fit
def rect(x, m):
    return m*x

def plot_fit(x, y, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_aspect('equal')
    maxplot = np.max([np.max(y), np.max(x)])
    ax.set_xlim(0, 1.1*maxplot)
    ax.set_ylim(0, 1.1*maxplot)
    x_points = np.linspace(0, 1.1*maxplot)
    ax.plot(x_points, x_points, '--')
    p, _ = curve_fit(rect, x, y )
    plt.plot(x_points, rect(x_points, *p), label=f'm={p[0]:.2f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

seed = 1 
dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
                {'dataset': 'CSRF19_V1',
                'neuronal_data_files': ['/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632669014376.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632932714885.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3633364677437.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634055946316.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634142311627.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634658447291.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634744023164.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635178040531.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635949043110.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636034866307.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636552742293.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637161140869.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637248451650.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637333931598.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637760318484.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637851724731.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638367026975.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638456653849.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638885582960.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638373332053.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638541006102.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638802601378.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638973674012.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639060843972.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639406161189.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640011636703.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639664527524.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639492658943.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639749909659.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640095265572.pickle',
                '/project/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle'],
                'image_cache_path': '/project/data/monkey/toliaslab/CSRF19_V1/images',
                'crop': 70,
                'subsample': 1,
                'seed': 1000,
                'time_bins_sum': 12,
                'batch_size': 128})

def create_random_pos_ensemble_model():
    model_m = copy.deepcopy(v1_convnext_ensemble)
    random_pos = torch.rand(1, 458, 1, 2)*.8-.4
    for model in model_m.members:
        model.readout['all_sessions'].mu.data += random_pos 
    return model_m

class MultipliedNeuronsModel(nn.Module):
    def __init__(self, factor=1, single_neuron_norm_factor=None, n=5, idxs=np.arange(458)):
        super().__init__()
        self.models = nn.ModuleList([create_random_pos_ensemble_model() for _ in range(n)])
        if single_neuron_norm_factor!=None:
            self.register_buffer('single_neuron_norm_factor', torch.cat([single_neuron_norm_factor for i in range(n)]).reshape(1, -1))
        else: 
            self.single_neuron_norm_factor = None
        self.factor = factor
        self.idxs = idxs
        self.n = n
    
    def forward(self, x):
        if self.single_neuron_norm_factor!=None:
            resp = torch.cat([m(x)[:, self.idxs] for m in self.models], -1)
            resp = resp/self.single_neuron_norm_factor
            resp = resp.pow(self.factor)
        else: 
            resp = torch.cat([m(x)[:, self.idxs] for m in self.models], -1)
            resp = resp.pow(self.factor)
        return resp

class SelectedNeuronsModel(nn.Module):
    def __init__(self, model, idxs):
        super().__init__()
        self.idxs = idxs
        self.model = model
    
    def forward(self, x):
        return self.model(x)[:, self.idxs]


dataloaders = get_data(dataset_fn, dataset_config)


outpainted = pickleread('/project/monkey_outpainted.pkl')
print(sorted(outpainted.keys()))

d = pickleread('/project/experiment_data/convnext/data_final_1_opt.pickle')
d2 = pickleread('/project/experiment_data/convnext/data_final_2_opt.pickle')
d.update(d2)
#%%
#%%
n=1
# idxs = list(d.keys())
idxs = np.arange(458)

model = SelectedNeuronsModel(v1_convnext_ensemble, idxs)
# model = v1_convnext_ensemble
# model = MultipliedNeuronsModel(n=n, idxs=idxs)
# model = v1_convnext.core
model.cuda()



#%%
def get_resps(model, out_imgs, opt_imgs, use_mask=True, fix_c=None):
    out_resps = []
    opt_resps = []
    with torch.no_grad():
        for idx in range(len(opt_imgs)):
            out_img = out_imgs[idx].reshape(1,1,93,93)
            opt_img = opt_imgs[idx].reshape(1,1,93,93)
            if use_mask:
                mask = torch.Tensor(create_mask_from_mei(opt_img, zscore_thresh=1.5, closing_iters=10)[0]).cuda().reshape(1,1,93,93)
                
                out_img = out_img*mask
                # plt.imshow(opt_img.cpu().squeeze(), vmin=-.6, vmax=.6, cmap='Greys_r')
                # plt.show()
                
                opt_img = opt_img*mask
                if fix_c != None:
                    out_img = fix_c[idx](out_img)
                    opt_img = fix_c[idx](opt_img)
                    
                # plt.imshow(opt_img.cpu().squeeze(), vmin=-.6, vmax=.6, cmap='Greys_r')
                # plt.show()
                # plt.imshow(out_img.cpu().squeeze(), vmin=-.6, vmax=.6, cmap='Greys_r')
                # plt.show()
            out_resps.append(model(out_img).squeeze())
            opt_resps.append(model(opt_img).squeeze())
        out_resps = torch.stack(out_resps).cpu()
        opt_resps = torch.stack(opt_resps).cpu()
    return out_resps, opt_resps

def get_corr_coeff(out_resps, opt_resps):
    with torch.no_grad():
        corr_coeff = [
            torch.corrcoef(torch.stack([out_resps[idx], opt_resps[idx]]))[0,1]
            for idx in range(len(out_resps))]
    return corr_coeff

# #%%
# resps = {}
# for idx in range(458):
#     resps[idx] = []
# for batch in tqdm(dataloaders['train']['all_sessions']):
#     x, y, e = batch
#     for i in range(len(y)):
#         for idx in range(458):
#             if e[i, idx] == True:
#                 resps[idx].append(float(y[i, idx].detach().cpu()))

# mean_r = np.array([np.mean(resps[idx]) for idx in range(458)])[idxs].reshape(1, -1)
# std_r = np.array([np.std(resps[idx]) for idx in range(458)])[idxs].reshape(1, -1)


#%%

# out_exc_resps, exc_resps = get_resps(model, out_imgs, exc_imgs, use_mask=True)
# out_exc_resps = (out_exc_resps-mean_r)/std_r
# exc_resps = (exc_resps-mean_r)/std_r
# out_inh_resps, inh_resps = get_resps(model, out_imgs, inh_imgs, use_mask=True)
# out_inh_resps = (out_inh_resps-mean_r)/std_r
# inh_resps = (inh_resps-mean_r)/std_r

# # out_exc_resps = out_exc_resps.T
# # exc_resps = exc_resps.T

# # out_inh_resps = out_inh_resps.T
# # inh_resps = inh_resps.T

# exc_corr_coeff = get_corr_coeff(out_exc_resps, exc_resps)
# inh_corr_coeff = get_corr_coeff(out_inh_resps, inh_resps)

# fig, ax= plt.subplots()
# ax.scatter(exc_corr_coeff, inh_corr_coeff, color='k', s=8)
# ax.set_aspect('equal')
# ax.set_xlabel('excitatory to outpainted correlation')
# ax.set_ylabel('inhibitory to outpainting correlation')
# ax.plot([0,1], [0,1], 'k--')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# # ax.set_title('method 2')
# plt.show()

# print(np.mean(exc_corr_coeff))
# print(np.mean(inh_corr_coeff))

#%%
outpainted = pickleread('/project/monkey_outpainted.pkl')
idxs = list(outpainted.keys())
idxs = sorted([int(i) for i in idxs])
contrast_surround_fix_list_exc = [ops.ChangeSurroundStd(torch.Tensor(d[idx]['exc_full_surr']).cuda(), torch.Tensor(d[idx]['mask_mei']).cuda(), std=(0, .1)) for idx in idxs]
contrast_surround_fix_list_inh = [ops.ChangeSurroundStd(torch.Tensor(d[idx]['inh_full_surr']).cuda(), torch.Tensor(d[idx]['mask_mei']).cuda(), std=(0, .1)) for idx in idxs]

#%%

idxs_outpainted = outpainted.keys()
out_imgs = [torch.Tensor(outpainted[idx][3:].mean(0).reshape(1,1,93,93)).cuda() for idx in idxs_outpainted]
exc_imgs = [torch.Tensor(d[int(idx)]['exc_full_surr']).reshape(1,1,93,93).cuda() for idx in idxs_outpainted]
inh_imgs = [torch.Tensor(d[int(idx)]['inh_full_surr']).reshape(1,1,93,93).cuda() for idx in idxs_outpainted]

exc_corr_coeff_list = []
inh_corr_coeff_list = []


# mean_r = np.array(pickleread('/project/mean_r.pickle'))[idxs].reshape(1,-1).repeat(n, 1)
# # mean_r = 0
# std_r = np.array(pickleread('/project/std_r.pickle'))[idxs].reshape(1,-1).repeat(n, 1)

for seed in np.arange(3,10):
    out_imgs = [torch.Tensor(outpainted[idx][seed].reshape(1,1,93,93)).cuda() for idx in idxs_outpainted]
    out_exc_resps, exc_resps = get_resps(model, out_imgs, exc_imgs, use_mask=True, fix_c = contrast_surround_fix_list_exc )
    out_inh_resps, inh_resps = get_resps(model, out_imgs, inh_imgs, use_mask=True, fix_c = contrast_surround_fix_list_inh)

    # out_exc_resps = out_exc_resps.flatten(1)
    # exc_resps = exc_resps.flatten(1)
    # out_inh_resps = out_inh_resps.flatten(1)
    # inh_resps = inh_resps.flatten(1)

    resps = torch.cat([out_exc_resps, exc_resps, out_inh_resps, inh_resps])
    mean_r = resps.mean(dim=0, keepdim=True)
    std_r = resps.std(dim=0,  keepdim=True)

    out_exc_resps = (out_exc_resps-mean_r)/std_r
    exc_resps = (exc_resps-mean_r)/std_r
    out_inh_resps = (out_inh_resps-mean_r)/std_r
    inh_resps = (inh_resps-mean_r)/std_r
    
    exc_corr_coeff = torch.stack(get_corr_coeff(out_exc_resps, exc_resps))
    inh_corr_coeff = torch.stack(get_corr_coeff(out_inh_resps, inh_resps))
    # fig, ax= plt.subplots()
    # ax.scatter(exc_corr_coeff, inh_corr_coeff, color='k', s=8)
    # ax.set_aspect('equal')
    # ax.set_xlabel('excitatory to outpainted correlation')
    # ax.set_ylabel('inhibitory to outpainting correlation')
    # ax.plot([0,1], [0,1], 'k--')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # # ax.set_title('method 2')
    # plt.show()
    # plot_fit(exc_corr_coeff.numpy(), inh_corr_coeff.numpy())
    exc_corr_coeff_list.append(exc_corr_coeff)
    inh_corr_coeff_list.append(inh_corr_coeff)
    # break

exc_corr_coeff =torch.stack(exc_corr_coeff_list).mean(dim=0)
inh_corr_coeff =torch.stack(inh_corr_coeff_list).mean(dim=0)

# plot_fit(exc_corr_coeff.numpy(), inh_corr_coeff.numpy())
fig, ax= plt.subplots()
ax.scatter(exc_corr_coeff, inh_corr_coeff, color='k', s=8)
ax.set_aspect('equal')
ax.set_xlabel(f'excitatory to outpainted correlation (mean={torch.mean(exc_corr_coeff):.2f})')
ax.set_ylabel(f'inhibitory to outpainting correlation (mean={torch.mean(inh_corr_coeff):.2f})')
ax.plot([0,1], [0,1], 'k--')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
# ax.set_title('method 2')
plt.show()

print(torch.mean(exc_corr_coeff))
print(torch.mean(inh_corr_coeff))


#%%
out_exc_resps, exc_resps = get_resps(model, out_imgs, exc_imgs, use_mask=True)
out_inh_resps, inh_resps = get_resps(model, out_imgs, inh_imgs, use_mask=True)

# out_exc_resps = out_exc_resps.T
# exc_resps = exc_resps.T

# out_inh_resps = out_inh_resps.T
# inh_resps = inh_resps.T

exc_corr_coeff = get_corr_coeff(out_exc_resps, exc_resps)
inh_corr_coeff = get_corr_coeff(out_inh_resps, inh_resps)

fig, ax= plt.subplots()
ax.scatter(exc_corr_coeff, inh_corr_coeff, color='k', s=5)
ax.set_aspect('equal')
ax.set_xlabel('excitatory to outpainted correlation')
ax.set_ylabel('inhibitory to outpainting correlation')
ax.plot([0,1], [0,1], 'k--')
plt.title('method 1')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
plt.show()
print(np.mean(exc_corr_coeff))
print(np.mean(inh_corr_coeff))

#%%

for iii in list(idxs_outpainted)[:10]:
    out_imgs_2 = [torch.Tensor(outpainted[iii][i].reshape(1,1,93,93)).cuda() for i in range(10)]
    fig, ax = plt.subplots(3,3,figsize=(20,20))
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(out_imgs_2[i*j + i].cpu().squeeze(), cmap='Greys_r', vmin=-.6, vmax=.6)
    # plt.tight_layout()
    plt.show()
    print(' ')


# %%
out_imgs.shape
# %%
idxs_outpainted
# %%

# %%
