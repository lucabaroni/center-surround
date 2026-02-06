#%%
from copyreg import pickle
from decimal import Clamped
from distutils.log import error
from random import gauss
from re import S
import torch
from classicalv1.GaborFilters import GaborFilter
from classicalv1.torchV1model import classicalV1, classicalV1random
import matplotlib.pyplot as plt
import numpy as np
from surroundmodulation.utils.misc import pickleread
import torch.nn as nn
from featurevis.utils import Compose
import torch.nn as nn
from surroundmodulation.analyses import *

#%%
population = classicalV1random(
    inputs_res=[93,93],
    inputs_fov=[2.67, 2.67],
    n_cells=10000,
    sf_values= 2.5, 
    sigma = [0.2],
    gabor_aspect_ratio=1, 
    simple_complex_ratio=1, 
)
population.cuda()
x = torch.randn(10, 1, 93,93).cuda()
x.shape

gbneuron = f = torch.Tensor(GaborFilter(
    pos=[0,0],
    sigma_x = 0.2,
    sigma_y = 0.267,
    sf = 2.5, 
    theta=np.pi/4,
    phase=0,
    res=[93,93],
    xlim= [-2.67/2, 2.67/2],
    ylim= [-2.67/2, 2.67/2]
))

class LN_model(nn.Module):
    def __init__(self, filter):
        super().__init__()
        self.register_buffer('filter', filter)
        self.elu = nn.ELU()

    def forward(self, x): 
        return self.elu(torch.einsum('bixy, xy->b', x, self.filter))+1

neuron = LN_model(gbneuron)

class Heeger_model(nn.Module):
    def __init__(self, population, neuron, gamma = 1, sigma = 1, p = 1, max_distance=1):
        super().__init__()
        self.population = population
        self.register_buffer('distance_from_center', self.get_distance_from_center())
        self.neuron = neuron
        self.sigma=sigma
        self.gamma = gamma
        self.max_distance = max_distance
        self.p = p

    def forward(self, x):
        y_i = self.neuron(x)
        y_pop = self.population(x).flatten(start_dim=1)
        num = self.gamma * (y_i)
        den = self.sigma + torch.mean(y_pop**self.p * (self.distance_from_center<self.max_distance).reshape(1,-1))#FIX
        return num/den
    
    def get_distance_from_center(self):
        pos = torch.Tensor(self.population.model.simple_cell_props['pos'])
        try:
            simple_cell_dist = torch.cdist(pos, torch.zeros_like(pos))[:,0]
        except:
            simple_cell_dist = torch.Tensor([])
        try:
            pos = torch.Tensor(self.population.model.complex_cells_props['pos'])
            complex_cell_dist = torch.cdist(pos, torch.zeros_like(pos))[:,0]
        except:
            complex_cell_dist = torch.Tensor([])
        dists = torch.cat([simple_cell_dist, complex_cell_dist])
        return dists
    

cmodel=Heeger_model(population=population, neuron=neuron, gamma=1, sigma=1, p=1, max_distance=10)
device = 'cuda'
cmodel.to(device)

mei = create_mei(cmodel, step_size=0.1, gaussianblur=1.)
mei = mei[0]
plt.imshow(mei.squeeze(), cmap='Greys_r')
plt.show()


mei_mask,  px_mask, py_mask = create_mask_from_mei(mei, zscore_thresh=1.5)
plt.imshow(mei_mask.squeeze(), cmap='Greys_r')
plt.show()

inh_full_surr, inh_only_surr, inh_full_surr_act, inh_only_surr_act = create_surround(cmodel, mei, mei_mask, objective='min', gaussianblur=1., device = device, surround_std=.10, step_size=0.1, num_iterations=3000)

plt.imshow(inh_full_surr.squeeze(), cmap='Greys_r')
plt.xticks([])
plt.yticks([])
plt.show()
plot_img(inh_full_surr, -.6,.6)

  # %%
plot_img(mei.squeeze(), -0.6,.6)

# %%
plt.imshow(inh_full_surr.squeeze(), vmin=-.6,vmax=.6, cmap='Greys_r')
plt.xticks([])
plt.yticks([])
plt.savefig('experiment_data/convnext/Heeger_divisive.pdf', dpi=300)
# %%
