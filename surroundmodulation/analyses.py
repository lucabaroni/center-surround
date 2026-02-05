#%%
from unicodedata import decimal
import torch
import numpy as np
import featurevis 
import featurevis.ops as ops
from featurevis.utils import Compose, Combine
from scipy import ndimage
from skimage import morphology
import random 
import imagen
from imagen.image import BoundingBox
from surroundmodulation.utils.misc import rescale
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from surroundmodulation.utils.misc import pickleread
from surroundmodulation.utils.plot_utils import plot_img
from surroundmodulation.utils.mask import find_mask_center
import math 
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.optimize import curve_fit



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_mei(model, std=0.05, seed=42, img_res = [93,93], pixel_min = -1.7876, pixel_max = 2.1919, gaussianblur=2., device=None, step_size=10, num_steps=1000):
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    initial_image = torch.randn(1, 1, *img_res, dtype=torch.float32).to(device)*std
    model.eval()
    model.to(device)
    initial_image = initial_image.to(device)
    
    # TODO decide if we need to add it 
    post_update =Compose([ops.ChangeStd(std), ops.ClipRange(pixel_min, pixel_max)])
    opt_x, fevals, reg_values = featurevis.gradient_ascent(
        model,
        initial_image, 
        step_size=step_size,
        num_iterations=num_steps, 
        post_update=post_update,
        gradient_f = ops.GaussianBlur(gaussianblur),
        print_iters=1001,
    )
    mei = opt_x.detach().cpu().numpy().squeeze()
    mei_act = fevals[-1]
    return mei,  mei_act


def find_mask_center(mask):
    # Create coordinate grids
    y, x = np.indices(mask.shape)

    # Calculate weighted centroids
    centroid_x = np.sum(x * mask) / np.sum(mask)
    centroid_y = np.sum(y * mask) / np.sum(mask)

    # print(f"Centroid: x={centroid_x}, y={centroid_y}")
    return (centroid_x, centroid_y)

def create_mask_from_mei(mei, zscore_thresh = 1.5, closing_iters=2, gaussian_sigma =1):
    params = {
        'zscore_thresh' : zscore_thresh,
        'closing_iters' : closing_iters, 
        'gaussian_sigma' : gaussian_sigma, 
    }   

    if type(mei) == type(torch.ones(1)):
        mei = mei.detach().cpu().numpy().squeeze()
    else: 
        mei = mei.squeeze()

    norm_mei = (mei - mei.mean()) / mei.std()

    thresholded = np.abs(norm_mei) > params['zscore_thresh']
    # Remove small holes in the thresholded image and connect any stranding pixels
    closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])
    #%%
    # Remove any remaining small objects
    labeled = morphology.label(closed, connectivity=2)
    most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    oneobject = labeled == most_frequent

    # Create convex hull just to close any remaining holes and so it doesn't look weird
    hull = morphology.convex_hull_image(oneobject)

    # Smooth edges
    smoothed = ndimage.gaussian_filter(hull.astype(np.float32),
                                        sigma=params['gaussian_sigma'])
    mask = smoothed  # final mask

    # Compute mask centroid
    px, py =  find_mask_center(mask)
    return mask, px, py

def create_surround(
    model,
    mei,
    mei_mask, 
    objective = 'max', 
    surround_std=0.05, 
    center_std = 0,
    img_res = [93,93], 
    pixel_min = -1.7876,
    pixel_max = 2.1919, 
    gaussianblur=2., 
    device=None,
    step_size = 10, 
    num_iterations = 1000,
    seed = 42,
    ):
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    sur_std = [center_std, surround_std]
    image_shape = (1, 1,*img_res)
    initial_image = torch.randn(image_shape, device=device)
    initial_image = initial_image * float(0.5) #LOOK INTO
    initial_image = initial_image
    model.to(device)
    model.eval()
    
    postup_op = Compose([ops.ChangeStd(float(surround_std)), ops.ClipRange(pixel_min, pixel_max)])
    if objective == 'max':
        gradient_f = ops.GaussianBlur(float(gaussianblur))
    elif objective == 'min':
        gradient_f = Compose([ops.GaussianBlur(float(gaussianblur)),ops.MultiplyBy(float(-1.))])
    else: 
        AssertionError('objective (of optimization) must be set to "min" or "max"')
    transform = ops.ChangeSurroundStd(mei, mei_mask, sur_std)

    only_surr, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                            post_update=postup_op,
                                                            gradient_f=gradient_f,
                                                            transform=transform,
                                                            step_size=step_size,
                                                            num_iterations=num_iterations, 
                                                            print_iters=1001)

    full_surr = transform(only_surr).squeeze().cpu().numpy()

    full_surr_act = fevals[-1]
    with torch.no_grad():
        only_surr_act = model(only_surr)

    # map to npy
    only_surr = only_surr.detach().squeeze().cpu().numpy()

    return full_surr, only_surr, full_surr_act, only_surr_act

def find_preferred_grating_parameters(
    all_neurons_model, 
    neuron_ids,
    orientations = np.linspace(0, np.pi, 37)[:-1],
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 0.2, 
    img_res = [93,93], 
    pixel_min = -1.7876, 
    pixel_max = 2.1919, 
    device = None,
    size = 2.35,
    ):
    if type(size) is int or type(size) is float: 
        size = [size]*2
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_neurons_model.to(device)
    all_neurons_model.eval()
    best_resp_dict = {}
    for idx in neuron_ids: 
        best_resp_dict[idx] = { 
            'resp_max':0
        }
    for orientation in tqdm(orientations):
        for sf in spatial_frequencies:
            for phase in phases: 
                grating = torch.Tensor(imagen.SineGrating(
                    orientation = orientation, 
                    frequency = sf,
                    phase = phase,
                    bounds = BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                    offset = 0,
                    scale = 1,
                    xdensity = img_res[1]/size[1],
                    ydensity = img_res[0]/size[0],
                )())
                grating = grating.reshape(1,*img_res).to(device)
                grating = rescale(grating, 0, 1, -1, 1)*contrast
                all_neuron_resp = all_neurons_model(grating.reshape(1,1,*img_res)).squeeze()
                for idx in neuron_ids:
                    if all_neuron_resp[idx]>best_resp_dict[idx]['resp_max']:
                        best_resp_dict[idx]['resp_max'] = all_neuron_resp[idx]
                        best_resp_dict[idx]['max_ori'] = orientation
                        best_resp_dict[idx]['max_phase'] = phase
                        best_resp_dict[idx]['max_sf']= sf
                        best_resp_dict[idx]['stim_max'] = grating
    return best_resp_dict

def find_preferred_masked_grating_parameters(
    model, 
    mei_mask, 
    orientations = np.linspace(0, np.pi, 37)[:-1],
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 0.2, 
    img_res = [93,93], 
    pixel_min = -1.7876, 
    pixel_max = 2.1919, 
    device = None,
    size = 2.35,
    ):
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    resp_max = 0
    if type(size) is int or type(size) is float: 
        size = [size]*2
    print(size)
    for orientation in orientations:
        for sf in spatial_frequencies:
            for phase in phases: 
                # call center-surround
                grating = torch.Tensor(imagen.SineGrating(
                    orientation = orientation, 
                    frequency = sf,
                    phase = phase,
                    bounds = BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                    offset = 0,
                    scale = 1,
                    xdensity = img_res[1]/size[1],
                    ydensity = img_res[0]/size[0],
                )())
                grating = grating.reshape(1,*img_res).to(device)
                grating = rescale(grating, 0, 1, 1, -1)*contrast #TODO fix contrast
                # grating = rescale(grating, -1, 1, pixel_min, pixel_max)
                grating = grating  * torch.Tensor(mei_mask).reshape(*grating.shape).to(device)
                resp=model(grating.reshape(1,1,*img_res))
                if resp>resp_max:
                    resp_max = resp
                    max_ori = orientation
                    max_phase = phase
                    max_sf = sf
                    stim_max = grating
    return max_ori, max_sf, max_phase, stim_max, resp_max

def get_offset_in_degr(x_pix, x_size_in_pix, x_size_in_deg):
    x_deg = ((x_pix - x_size_in_pix/2) / (x_size_in_pix/2)) * x_size_in_deg/2
    return x_deg

def size_tuning_all_phases(
    model, 
    x_pix, 
    y_pix, 
    preferred_ori,
    preferred_sf, 
    contrast=0.2, 
    radii = np.linspace(0.05, 2, 40), 
    phases=np.linspace(0, 2*np.pi, 37)[:-1],
    return_all=False,
    pixel_min = -1.7876,
    pixel_max = 2.1919, 
    device = None,
    size = 2.35,
    img_res = [93,93]
    ):
    if type(size) is int or type(size) is float: 
        size = [size]*2
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    resp = np.zeros([len(radii), len(phases)])
    gratings = np.zeros([len(radii), len(phases), *img_res])
    with torch.no_grad():
        for i, radius in enumerate(radii):
            for j, phase in enumerate(phases):
                a = imagen.SineGrating(orientation= preferred_ori,
                                        frequency= preferred_sf,
                                        phase= phase,
                                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                        offset = 0, 
                                        scale=1 ,
                                        xdensity = img_res[1]/size[1],
                                        ydensity = img_res[0]/size[0])()

                b = imagen.Constant(scale=0.5,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity = img_res[1]/size[1],
                                ydensity = img_res[0]/size[0])()

                c = imagen.Disk(smoothing=0.0,
                                size=radius*2,
                                scale=1.0,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity = img_res[1]/size[1],
                                ydensity = img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()    
                
                d1 = np.multiply(a,c)
                d2 = np.multiply(b,-(c-1.0))
                d =  np.add.reduce([d1,d2])
                d = torch.Tensor(d).reshape(1, 1, *img_res).to(device)
                input = rescale(d, 0, 1, -1, 1)*contrast
                # input = rescale(input, -1, 1, pixel_min, pixel_max)
                resp[i, j] = model(input)
                gratings[i, j] = input.squeeze().reshape(*img_res).cpu().numpy()
                # TODO improve
                idx = np.unravel_index(resp.argmax(), [len(radii), len(phases)])
                top_grating = gratings[idx]
                top_radius = radii[idx[0]]
                top_phase = phases[idx[1]]
                top_resp = resp[idx]
    if return_all:
        return top_radius, top_phase, top_grating, top_resp, resp, gratings
    else: 
        return top_radius, top_phase, top_grating, top_resp

def get_orientation_contrast_stim(
    x_pix, 
    y_pix, 
    center_radius, 
    center_orientation,
    spatial_frequency, 
    phase, 
    surround_radius, 
    surround_orientation, 
    gap, 
    contrast, 
    size, 
    num_pix_x,
    num_pix_y, 
):
    if type(size) is int or type(size) is float: 
        size = [size]*2
    center = imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=center_radius*2.0),
                                orientation=center_orientation,
                                frequency=spatial_frequency,
                                phase=phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=num_pix_x/size[1],
                                ydensity=num_pix_y/size[0], 
                                x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                                y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()

    r = (center_radius + surround_radius + gap)/2.0
    t = (surround_radius - center_radius - gap)/2.0
    surround = imagen.SineGrating(mask_shape=imagen.Ring(thickness=t*2.0, smoothing=0.0, size=r*2.0),
                                    orientation=surround_orientation + center_orientation,
                                    frequency=spatial_frequency,
                                    phase=phase,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                    offset = -1,
                                    scale=2,   
                                    xdensity=num_pix_x/size[1],
                                    ydensity=num_pix_y/size[0],
                                    x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                                    y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()

    center_disk = (imagen.Disk(smoothing=0.0,
                            size=center_radius*2.0, 
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            xdensity=num_pix_x/size[1],
                            ydensity=num_pix_y/size[0],
                            x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                            y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()-1)*-1

    outer_disk = imagen.Disk(smoothing=0.0,
                            size=(center_radius+gap)*2,
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            xdensity=num_pix_x/size[1],
                            ydensity=num_pix_y/size[0],
                            x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                            y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()
    ring = center_disk*outer_disk
    x = np.add.reduce([np.maximum(center, surround), ring])*contrast
    return x 

def orientation_contrast(
    model, 
    x_pix, 
    y_pix, 
    preferred_ori,
    preferred_sf, 
    center_radius,
    orientation_diffs, 
    phases,
    gap = 0.2,
    contrast=0.2,
    img_res = [93,93], 
    pixel_min = -1.7876,
    pixel_max = 2.1919, 
    device = None,
    size = 2.35
    ):
    if type(size) is int or type(size) is float: 
        size = [size]*2
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    oc_stims = []
    oc_resps = []
    with torch.no_grad():
        for od in orientation_diffs:
            oc_stim = [get_orientation_contrast_stim(
                    x_pix=x_pix, 
                    y_pix=y_pix, 
                    center_radius = center_radius, 
                    center_orientation=preferred_ori,
                    spatial_frequency = preferred_sf, 
                    phase = phase, 
                    surround_radius = 100, 
                    surround_orientation = od, 
                    gap = gap,  
                    contrast = contrast, 
                    size = size, 
                    num_pix_x = img_res[1],
                    num_pix_y = img_res[0],
                ) for phase in phases]
            oc_stim = np.stack(oc_stim).reshape(-1, 1, *img_res)
            oc_stim = torch.Tensor(oc_stim).to(device)
            # oc_stim = rescale(oc_stim, -1,1, pixel_min, pixel_max)
            oc_resp = model(oc_stim)
            oc_stims.append(oc_stim.detach().cpu().numpy().squeeze())
            oc_resps.append(oc_resp.detach().cpu().numpy().squeeze())
        oc_stims = np.stack(oc_stims)
        oc_resps = np.stack(oc_resps)
    return oc_stims, oc_resps
        

def dot_stimulation(
    all_neurons_model,
    neuron_ids,  
    dot_size_in_pixels=4,
    contrast = 1, 
    num_dots=200000,
    img_res = [93,93], 
    pixel_min = -1.7876, 
    pixel_max = 2.1919, 
    device = None,
    bs = 40,
    seed = 0,
):
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate the tensors
    tensors = []
    resp = []
    
    all_neurons_model.to(device)
    with torch.no_grad():
        no_stim_resp= all_neurons_model(torch.ones(1,1,93,93, device=device)* (pixel_min + pixel_max)/2)
        for _ in range(num_dots):
            tensor = np.zeros(img_res, dtype=np.int)
            x = np.random.randint(0, img_res[0] - dot_size_in_pixels + 1)
            y = np.random.randint(0, img_res[1] - dot_size_in_pixels + 1)
            square_values = np.random.choice([-1, 1])
            tensor[x:x+dot_size_in_pixels, y:y+dot_size_in_pixels] = square_values*contrast
            tensors.append(tensor)
        
        # Convert the list of tensors to a NumPy array
        tensors_array = torch.Tensor(np.array(tensors))
        for i in tqdm(range(0, num_dots, bs)):
            model_input = rescale(tensors_array[i:i+bs], -1, 1, pixel_min, pixel_max).reshape(bs, 1, 93,93).to(device)
            resp.append(all_neurons_model(model_input))
        resp = torch.cat(resp, dim=0)

    
        all_neurons_outputs = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.Tensor(np.abs(tensors_array)).to(device),
                            torch.Tensor(resp-no_stim_resp).to(device)
                            )
    dot_stim_dict = {}     
    for idx in neuron_ids:
        dot_stim_dict[idx] = all_neurons_outputs[idx]
    return dot_stim_dict



# Function to translate (shift) the image
def translate_image(image, x, y):
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# Function to rotate the image
def rotate_image(image, angle):
    center = (image.shape[1]//2, image.shape[0]//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

def create_circular_mask(image_shape, x, y, radius):
    """
    Create a circular mask for an image.

    Parameters:
    - image_shape: tuple, shape of the image (height, width).
    - x, y: coordinates of the center of the circle.
    - radius: radius of the circle.

    Returns:
    - mask: numpy array, mask with a circle.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, (255), thickness=-1)
    return mask
    
def calculate_split_lines(position, orientation):
    """
    Calculate the equations of two orthogonal lines that define the split based on the position and orientation.

    Parameters:
    position (tuple): The position (x, y) where the lines intersect.
    orientation (float): The angle in degrees of one of the lines.

    Returns:
    tuple: Two lambda functions representing the equations of the lines.
    """
    # Convert orientation to radians
    theta = np.radians(orientation + 45)
    x0, y0 = position

    # Ensure tan(theta) is not zero to avoid division by zero in the orthogonal line
    if np.tan(theta) == 0:
        theta += 0.0001  # Slight adjustment to avoid zero

    # Line equations
    line1 = lambda x: (np.tan(theta) * (x - x0)) + y0
    line2 = lambda x: (-1/np.tan(theta) * (x - x0)) + y0

    return line1, line2

def create_quadrant_masks(shape, position, orientation):
    """
    Creates binary masks for each of the four quadrants in an image.

    Parameters:
    shape (tuple): The shape of the image (height, width).
    position (tuple): The position (x, y) where the lines intersect.
    orientation (float): The angle in degrees of one of the lines.

    Returns:
    tuple: Four binary masks corresponding to the four quadrants.
    """
    height, width = shape
    line1, line2 = calculate_split_lines(position, orientation)

    # Create masks for each quadrant
    mask1 = np.zeros((height, width), dtype=bool)
    mask2 = np.zeros((height, width), dtype=bool)
    mask3 = np.zeros((height, width), dtype=bool)
    mask4 = np.zeros((height, width), dtype=bool)

    for x in range(width):
        for y in range(height):
            if y < line1(x) and y > line2(x):
                mask1[y, x] = True
            elif y < line1(x) and y <= line2(x):
                mask2[y, x] = True
            elif y >= line1(x) and y > line2(x):
                mask3[y, x] = True
            else:  # y >= line1(x) and y <= line2(x)
                mask4[y, x] = True

    return mask1+mask4, mask3+mask2

def create_odd_gabor_filters(num_filters, orientation, frequency, nx, aspect_ratio, kernel_size=15, device='cuda'):
    """
    Create odd Gabor filters in PyTorch with a fixed orientation, varying phases, specified frequency, and sigmas based on nx and aspect ratio.

    Parameters:
    num_filters (int): Number of filters with different phases.
    orientation (float): The fixed orientation of all filters in degrees.
    frequency (float): The frequency of the sinusoidal component.
    nx (int): Number of nodes (cycles) of the sinusoidal wave along x-axis.
    aspect_ratio (float): Ratio of sigma_y to sigma_x.
    kernel_size (int): The size of the kernel (assumed to be square).
    device (str): Device to which the filters should be transferred ('cuda' or 'cpu').

    Returns:
    torch.Tensor: A tensor of Gabor filters.
    """
    # Convert orientation to radians
    orientation_rad = torch.tensor(math.radians(orientation))

    # Calculate sigma based on frequency and nx
    wavelength = 1 / frequency
    sigma_x = wavelength / (nx * math.sqrt(2 * math.pi))
    sigma_y = sigma_x * aspect_ratio

    # Prepare the grid for the kernel
    grid_val = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size, device=device)
    x, y = torch.meshgrid(grid_val, grid_val)
    x_theta = x * torch.cos(orientation_rad) + y * torch.sin(orientation_rad)
    y_theta = -x * torch.sin(orientation_rad) + y * torch.cos(orientation_rad)

    # Gaussian envelope (elongated)
    gaussian = torch.exp(-0.5 * ((x_theta**2 / sigma_x**2) + (y_theta**2 / sigma_y**2)))

    filters = []
    for phase in torch.linspace(0, 2 * math.pi, num_filters, device=device):
        # Sine wave with varying phase
        sinusoid = torch.sin(2 * math.pi * frequency * x_theta + phase)

        # Create the Gabor filter
        gabor_filter = gaussian * sinusoid
        filters.append(gabor_filter)

    return torch.stack(filters)

def analysis1(mei_masks, surround_meis, oris, angles, plot=False, sf=1/7, mean=0,aspect_ratio=2):
    conv = nn.Conv2d(
        in_channels = 1, 
        out_channels= 36, 
        kernel_size=31, 
        padding='same', 
    )
    with torch.no_grad():
        orth_q = []
        coll_q = []
        for idx in range(len(oris)):
            ori= oris[idx]
            input_img = surround_meis[idx]
            mask = mei_masks[idx]
            px, py = find_mask_center(mask)
            mask_orth, mask_coll=create_quadrant_masks([93,93], [px, py], -ori)
            neg_mask= (1-mask)>0.5
            mask_circle = create_circular_mask((93,93), int(px), int(py), 35)
            mask_orth = neg_mask*mask_orth * mask_circle
            mask_coll = neg_mask*mask_coll * mask_circle
            input_img = torch.tensor(input_img).reshape(1, 1, 93, 93)-mean
            orth = []
            coll = []

            for angle in angles:
                filters = create_odd_gabor_filters(num_filters=36, orientation=ori + angle, frequency=sf, nx=1, aspect_ratio = aspect_ratio, kernel_size=31, device='cpu')
                conv.weight.data=torch.Tensor(filters.reshape(36, 1, 31,31))
                conv.to('cuda')
                output = conv(input_img.to('cuda'))
                maxed_output = torch.max(output, dim=1)[0].squeeze()
                if plot==True:
    
                    plt.imshow(filters[0])
                    plt.show()
                    plt.imshow(maxed_output.cpu().detach().numpy())
                    plt.show()
                    plt.imshow(maxed_output.cpu().detach().numpy()*mask_orth)
                    plt.show()
                    plt.imshow(maxed_output.cpu().detach().numpy()*mask_coll)
                    plt.show()
                orth.append(maxed_output.cpu().detach().numpy()[mask_orth>0.5].mean())
                coll.append(maxed_output.cpu().detach().numpy()[mask_coll>0.5].mean())
            orth_q.append(np.array(orth))
            coll_q.append(np.array(coll))
        orth_q = np.stack(orth_q)
        coll_q = np.stack(coll_q)
    return coll_q, orth_q

def analisys2(mei_masks, surround_meis, oris, angles = [0, 90], sf=1/7, aspect_ratio=2):
    output_list = []
    with torch.no_grad():
        for angle in angles:
            conv_c = nn.Conv2d(
                in_channels = 1, 
                out_channels= 36, 
                kernel_size=31, 
                padding='valid', 
            ) 
            filters = create_odd_gabor_filters(num_filters=36, orientation=90 + angle, frequency=sf, nx=1, aspect_ratio = aspect_ratio, kernel_size=31, device='cpu')
            conv_c.weight.data=torch.Tensor(filters.reshape(36, 1, 31,31))
            conv_c.to('cuda')
            coll = []
            for idx in range(len(oris)):
                ori = oris[idx] 
                input_img = surround_meis[idx]
                # plot_img(input_img, pixel_min, pixel_max)
                mask = mei_masks[idx]

                px, py = find_mask_center(mask)
                image = np.array(input_img).squeeze()

                angle = 90-ori
                tpx = (93/2)-px
                tpy = (93/2)-py
                # Translate the image
                translated_image = translate_image(image, tpx, tpy)
        

                # Rotate the image
                rotated_image = rotate_image(translated_image, angle)
            
                # plt.imshow(rotated_image)
                # plt.show()
                angles = np.linspace(0, 180, 3)[:-1]
                input_img = torch.tensor(rotated_image).reshape(1,1, 93,93)

                output_c = conv_c(input_img.to('cuda'))
                maxed_output_c = torch.max(output_c, dim=1)[0].squeeze()  
                coll.append(maxed_output_c)
            output_list.append(torch.stack(coll).mean(dim=0))
    return output_list

def find_extremes_of_mask(mask):
    l0, l1 = mask.shape
    mask = mask>0.5
    dim_0 = mask.max(axis=1)
    first_0, last_0 = find_first_and_last_true(dim_0)
    dim_1 = mask.max(axis=0)
    first_1, last_1 = find_first_and_last_true(dim_1)
    return first_0, l0+last_0 , first_1, l1+last_0 +last_1 

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

def get_cardinal_masks(mask):
    mask = mask>0.5
    first_0, last_0, first_1, last_1 = find_extremes_of_mask(mask)
    top = np.zeros_like(mask)
    bottom = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)

    top[first_0-16:first_0-1, first_1:last_1] = 1
    bottom[last_0+2:last_0+17, first_1:last_1] = 1
    left[first_0:last_0, first_1-16:first_1-1] = 1
    right[first_0:last_0, last_1+2:last_1+17,] = 1
    
    return top, bottom, left, right

def analisys3(mei_masks, surround_meis, oris, angles = [0, 90], sf=1/7, aspect_ratio=2):
    all_outputs = []
    collinear_region_outputs = []
    orthogonal_region_outputs = []
    with torch.no_grad():
        for angle in angles:
            conv_c = nn.Conv2d(
                in_channels = 1, 
                out_channels= 36, 
                kernel_size=31, 
                padding='same', 
            ) 
            filters = create_odd_gabor_filters(num_filters=36, orientation=90 + angle, frequency=sf, nx=1, aspect_ratio = aspect_ratio, kernel_size=31, device='cpu')
            conv_c.weight.data=torch.Tensor(filters.reshape(36, 1, 31,31))
            conv_c.to('cuda')
            coll = []

            collinear_region_single_angle = []
            orthogonal_region_single_angle = []

            centered_masks = []
            for idx in range(len(surround_meis)): #this loops over images
                ori = oris[idx] 
                input_img = surround_meis[idx]
                # plot_img(input_img, pixel_min, pixel_max)
                mask = mei_masks[idx]

                px, py = find_mask_center(mask)
                image = np.array(input_img).squeeze()

                angle = 90-ori
                tpx = (93/2)-px
                tpy = (93/2)-py
                # Translate the image
                translated_image = translate_image(image, tpx, tpy)
                translated_mask = translate_image(mask, tpx, tpy)

                # Rotate the image
                rotated_image = rotate_image(translated_image, angle)
                rotated_mask = rotate_image(translated_mask, angle) > 0.5
                centered_masks.append(rotated_mask)
                
                top, bottom, left, right = get_cardinal_masks(rotated_mask)

                collinear_region = top + bottom
                orthogonal_region = left + right

                # plt.imshow(rotated_image)
                # plt.show()
                angles = np.linspace(0, 180, 3)[:-1]
                input_img = torch.tensor(rotated_image).reshape(1,1, 93,93)

                output_c = conv_c(input_img.to('cuda'))
                maxed_output_c = torch.max(output_c, dim=1)[0].squeeze()  
                coll.append(maxed_output_c)

                collinear_region_single_angle.append(maxed_output_c.cpu().numpy()*collinear_region)
                orthogonal_region_single_angle.append(maxed_output_c.cpu().numpy()*orthogonal_region)

            all_outputs.append(torch.stack(coll).cpu().numpy()) # this is averaging across neurons
            collinear_region_outputs.append(np.stack(collinear_region_single_angle))
            orthogonal_region_outputs.append(np.stack(orthogonal_region_single_angle))
            # output_list.append(torch.stack(coll).mean(dim=0)) # this is averaging across neurons
    return all_outputs, collinear_region_outputs, orthogonal_region_outputs, centered_masks

#%% dot stim fitting
def gaussian2D_with_correlation(xy, A, x0, y0, sigma_x, sigma_y, rho):
    x, y = xy
    a = 1.0 / (2 * (1 - rho**2))
    b = ((x - x0)**2) / (sigma_x**2)
    c = 2 * rho * (x - x0) * (y - y0) / (sigma_x * sigma_y)
    d = ((y - y0)**2) / (sigma_y**2)
    return A * np.exp(-a * (b - c + d))

def gauss_fit(dot_stim_img, img_size=[93,93]): 
    data = np.clip(dot_stim_img, a_min=0, a_max=None)
    x_data = np.arange(0, img_size[1])
    y_data = np.arange(0, img_size[0])
    x, y = np.meshgrid(x_data, y_data)

    # Flatten for fitting
    x_data, y_data = x.ravel(), y.ravel()
    z_data = data.ravel()

    # Initial guess [A, x0, y0, sigma_x, sigma_y, rho]
    initial_guess = [500, 93/2, 93/2, 10, 10, 0]

    # Fit the model
    params, covariance = curve_fit(gaussian2D_with_correlation, (x_data, y_data), z_data, p0=initial_guess)

    # Extract the optimized parameters
    A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt = params
    print(f"Optimized parameters: A = {A_opt}, x0 = {x0_opt}, y0 = {y0_opt}, sigma_x = {sigma_x_opt}, sigma_y = {sigma_y_opt}, rho = {rho_opt}")

    fitted_data = gaussian2D_with_correlation((x_data, y_data), *params)
    return A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_data.reshape(*img_size)

# %% size tuning

def find_first_satisfying_value(array, condition_func):
    for index, value in enumerate(array):
        if condition_func(value):
            return index, value
    return None  # or a default value if no value satisfies the condition

def assert_monotonic_increasing_until_index(array, index):
    # Ensure the index is within the bounds of the array
    if index >= len(array):
        raise ValueError("Index is out of the array's bounds.")
    
    for i in range(index):
        if array[i] > array[i + 1]:
            return False  # The array is not monotonically increasing up to the given index
    return True  # The array is monotonically increasing up to the given index

def assert_size_tuning(radii, st_r, remove_baseline=True):
    assert radii[0] ==0 
    if remove_baseline:
        st_r = st_r- st_r[0]

    st_r_max = st_r.max()
    st_r_argmax = st_r.argmax()


    # print(st_r)
    condition_func = lambda x: x > .95*st_r_max  # Condition: value is greater than 3
    st_r_95_idx, st_r_95 = find_first_satisfying_value(st_r, condition_func)
    monotonic = assert_monotonic_increasing_until_index(st_r, st_r_argmax)
    
    if monotonic: 
        suppression_idx = 1-(st_r[-1]-st_r[0])/(st_r_max-st_r[0])
    else:
        suppression_idx=None

    return monotonic, radii[st_r_95_idx], radii[st_r_argmax], suppression_idx, st_r_95
# #%%
# orientation =  np.pi
# sf = 7
# phase = 0
# size = 2.35
# img_res = [93,93]
# grating = torch.Tensor(imagen.SineGrating(
#                     orientation = orientation, 
#                     frequency = sf,
#                     phase = phase,
#                     bounds = BoundingBox(radius=size/2),
#                     offset = 0,
#                     scale = 1,
#                     xdensity = img_res[0]/size, #fix this
#                     ydensity = img_res[1]/size,
#                 )().reshape(1, *img_res)).to(device)
# plot_img(grating, 0, 1)

# # %%
# orientation =  np.pi
# sf = np.linspace(1, 9, 33)
# phase = 0
# size = 2.35
# img_res = [93,93]
# grating = torch.Tensor(imagen.SineGrating(
#                     orientation = orientation, 
#                     frequency = sf,
#                     phase = phase,
#                     bounds = BoundingBox(radius=size/2),
#                     offset = 0,
#                     scale = 1,
#                     xdensity = img_res[0]/size, #fix this
#                     ydensity = img_res[1]/size,
#                 )().reshape(1, *img_res)).to(device)
# plot_img(grating, 0, 1)

# # %%

# # %%
# np.linspace(1, 8, 29)
# # %%

# preferred_ori = 0
# preferred_sf = 4
# radius = 0.2
# x_pix = 43
# y_pix = 40
# a = imagen.SineGrating(orientation= preferred_ori,x
#                         frequency= preferred_sf,
#                         phase= phase,
#                         bounds=BoundingBox(radius=size/2),
#                         offset = 0, 
#                         scale=1 ,
#                         xdensity= img_res[0]/size,
#                         ydensity= img_res[1]/size)()

# b = imagen.Constant(scale=0.5,
#                 bounds=BoundingBox(radius=size/2),
#                 xdensity= img_res[0]/size,
#                 ydensity= img_res[1]/size)()

# c = imagen.Disk(smoothing=0.0,
#                 size=radius*2,
#                 scale=1.0,
#                 bounds=BoundingBox(radius=size/2),
#                 xdensity=img_res[0]/size,
#                 ydensity=img_res[1]/size, 
#                 x = get_offset_in_degr(x_pix, img_res[0], size), 
#                 y = -get_offset_in_degr(y_pix, img_res[1], size))()    

# d1 = np.multiply(a,c)
# d2 = np.multiply(b,-(c-1.0))
# d =  np.add.reduce([d1,d2])
# d = torch.Tensor(d).reshape(1, 1, 93,93).to(device)
# plot_img(d, 0,1)
# # %%

# def get_orientation_contrast_stim(
#     x_pix, 
#     y_pix, 
#     center_radius, 
#     center_orientation,
#     spatial_frequency, 
#     phase, 
#     surround_radius, 
#     surround_orientation, 
#     gap, 
#     contrast, 
#     size, 
#     num_pix_x,
#     num_pix_y, 
# ):
#     center = imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=center_radius*2.0),
#                                 orientation=center_orientation,
#                                 frequency=spatial_frequency,
#                                 phase=phase,
#                                 bounds=BoundingBox(radius=size/2.0),
#                                 offset = -1,
#                                 scale=2,  
#                                 xdensity=num_pix_x/size,
#                                 ydensity=num_pix_y/size, 
#                                 x = get_offset_in_degr(x_pix, num_pix_x, size), 
#                                 y = -get_offset_in_degr(y_pix, num_pix_y, size))()

#     r = (center_radius + surround_radius + gap)/2.0
#     t = (surround_radius - center_radius - gap)/2.0
#     surround = imagen.SineGrating(mask_shape=imagen.Ring(thickness=t*2.0, smoothing=0.0, size=r*2.0),
#                                     orientation=surround_orientation + center_orientation,
#                                     frequency=spatial_frequency,
#                                     phase=phase,
#                                     bounds=BoundingBox(radius=size/2.0),
#                                     offset = -1,
#                                     scale=2,   
#                                     xdensity=num_pix_x/size,
#                                     ydensity=num_pix_y/size,
#                                     x = get_offset_in_degr(x_pix, num_pix_x, size), 
#                                     y = -get_offset_in_degr(y_pix, num_pix_y, size))()

#     center_disk = (imagen.Disk(smoothing=0.0,
#                             size=center_radius*2.0, 
#                             bounds=BoundingBox(radius=size/2.0),
#                             xdensity=num_pix_x/size,
#                             ydensity=num_pix_y/size,
#                             x = get_offset_in_degr(x_pix, num_pix_x, size), 
#                             y = -get_offset_in_degr(y_pix, num_pix_y, size))()-1)*-1

#     outer_disk = imagen.Disk(smoothing=0.0,
#                             size=(center_radius+gap)*2,
#                             bounds=BoundingBox(radius=size/2.0),
#                             xdensity=num_pix_x/size,
#                             ydensity=num_pix_y/size,
#                             x = get_offset_in_degr(x_pix, num_pix_x, size), 
#                             y = -get_offset_in_degr(y_pix, num_pix_y, size))()
#     ring = center_disk*outer_disk
#     x = np.add.reduce([np.maximum(center, surround), ring])*contrast
#     return x 

# d = get_orientation_contrast_stim(
#     x_pix=x_pix, 
#     y_pix=y_pix, 
#     center_radius=0.5, 
#     center_orientation=0, 
#     spatial_frequency=4,
#     phase=0,
#     surround_radius=100, 
#     surround_orientation=5,
#     gap=0.2, 
#     contrast=1, 
#     size=2.35,
#     num_pix_x=93,
#     num_pix_y=93
# )
# plot_img(d, -1,1)

# orientation = np.pi/2
# sf = 0.03
# phase = np.pi/2
# size = [67.5, 120]
# img_res = [36, 64]
# grating = torch.Tensor(imagen.SineGrating(
#                     orientation = orientation, 
#                     frequency = sf,
#                     phase = phase,
#                     bounds = BoundingBox(
#                         points=(
#                             (-size[1]/2, -size[0]/2),
#                             (size[1]/2, size[0]/2)
#                             )
#                         ), 
#                     offset = 0,
#                     scale = 1,
#                     xdensity = img_res[1]/size[1], 
#                     ydensity = img_res[0]/size[0],
#                 )().reshape(1, *img_res)).to(device)
# plot_img(grating, 0, 1)

# print(120/64, 36*1.875)
# %%
# gap = 10
# radii = np.linspace(4, 70, 28)
# d = get_orientation_contrast_stim(
#     x_pix=30,
#     y_pix=15, 
#     center_radius=, 
#     center_orientation=0, 
#     spatial_frequency=0.06,
#     phase=0,
#     surround_radius=100, 
#     surround_orientation=5,
#     gap=5, 
#     contrast=1, 
#     size=[67.5,120],
#     num_pix_x=64,
#     num_pix_y=36,
# )
# plot_img(d, -1,1)
# %%

# %%
