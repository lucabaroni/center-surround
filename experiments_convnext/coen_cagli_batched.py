# %%

import os
from PIL import Image
import numpy as np
import torch
from surroundmodulation.utils.misc import pickleread, picklesave
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
import gc  

#%%
def apply_circular_filter(images, position, size_in_degrees, return_filter=False):
    """
    Apply a circular filter to a batch of images.

    Args:
        images (np.ndarray or torch.Tensor): Input images, shape (B, H, W) or (H, W).
        position (tuple of floats): Center of the circle (x, y) in pixel coordinates.
        size_in_degrees (float): Diameter of the circle in degrees.
        return_filter (bool): Whether to return the filter along with the filtered images.

    Returns:
        filtered_images (np.ndarray): Images with the circular filter applied, shape (B, H, W).
        filter (np.ndarray, optional): The circular filter (if return_filter=True), shape (H, W).
    """
    # Handle single image input by adding batch dimension
    if isinstance(images, Image.Image):
        images = np.array(images, dtype=np.float32)
    if isinstance(images, np.ndarray) and images.ndim == 2:
        images = images[None, ...]
    elif isinstance(images, torch.Tensor) and images.ndim == 2:
        images = images.unsqueeze(0)
    # Now images is (B, H, W)
    if isinstance(images, torch.Tensor):
        images_np = images.cpu().numpy()
    else:
        images_np = images

    B, H, W = images_np.shape
    size_in_pixel_scale = H
    half_degr_in_pixel_scale = size_in_pixel_scale / size_in_degrees * 0.5

    y, x = np.ogrid[:H, :W]
    center_x, center_y = position
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    circular_filter = (distance <= half_degr_in_pixel_scale).astype(np.float32)

    # Apply the filter to each image in the batch
    filtered_images = images_np * circular_filter[None, :, :]

    # Return in the same type as input
    if isinstance(images, torch.Tensor):
        filtered_images = torch.from_numpy(filtered_images).to(images.device).type_as(images)
        circular_filter_out = torch.from_numpy(circular_filter).to(images.device).type_as(images)
    else:
        circular_filter_out = circular_filter

    if return_filter:
        return filtered_images, circular_filter_out
    return filtered_images



def calculate_crop_box(image, crop_size, crop_position="center"):
    width, height = image.size
    target_width, target_height = crop_size

    if crop_position == "center":
        left = (width - target_width) // 2
        upper = (height - target_height) // 2
    elif crop_position == "top_left":
        left, upper = 0, 0
    elif crop_position == "top_right":
        left, upper = width - target_width, 0
    elif crop_position == "bottom_left":
        left, upper = 0, height - target_height
    elif crop_position == "bottom_right":
        left, upper = width - target_width, height - target_height
    else:
        raise ValueError(f"Unsupported position: {crop_position}")

    right = left + target_width
    lower = upper + target_height
    return (left, upper, right, lower)

class ImageDataset(Dataset):
    def __init__(self, image_dir, crop_size=93, transform=None, pixel_min=-1.7876, pixel_max=2.1919, crop_position="center"):
        """
        Args:
            image_dir (str): Directory with all the images.
            crop_size (int): Size of the square crop to apply to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
            pixel_min (float): Minimum pixel value for rescaling.
            pixel_max (float): Maximum pixel value for rescaling.
        """
        super(ImageDataset, self).__init__()
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist.")
        if not os.path.isdir(image_dir):
            raise ValueError(f"Image directory {image_dir} is not a directory.")
        if crop_size <= 0:
            raise ValueError("Crop size must be a positive integer.")
        if not isinstance(crop_size, int):
            raise ValueError("Crop size must be an integer.")
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        if pixel_min >= pixel_max:
            raise ValueError("pixel_min must be less than pixel_max.")
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f"{str(i).zfill(6)}.png") for i in range(1,1001)]
        self.crop_position= crop_position

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")  # Convert to grayscale (1 channel)

        # Calculate crop box based on the desired crop size and position
        crop_box = calculate_crop_box(image, crop_size=(self.crop_size, self.crop_size), crop_position=self.crop_position)
        cropped_image = image.crop(crop_box)

        # Convert to NumPy array and rescale pixel values
        image_array = np.array(cropped_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image_array = image_array * (self.pixel_max - self.pixel_min) + self.pixel_min  # Rescale to [pixel_min, pixel_max]

        # Convert to tensor and add channel dimension
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor

def create_dataloader(image_dir, batch_size, crop_size=93, crop_position="center"):
    dataset = ImageDataset(image_dir=image_dir, crop_size=crop_size, crop_position=crop_position)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
#%%
data_dict = pickleread('/project/experiment_data/convnext/data_final_combined_opt.pickle') # check if this is the correct file
model = v1_convnext_ensemble
model.cuda()
# Example usage
image_dir = "/project/v1_data/images"
image_size_in_degrees = 2.67
pixel_min=-1.7876
pixel_max=2.1919
pixel_mean = (pixel_min + pixel_max) / 2
N = 1000
results = {}
# for index in data_dict.keys():
#     results[index] = {
#         'full_field_images': np.zeros((N, 93, 93)),
#         'full_field_responses': np.zeros(N),
#         'circluar_filter_images': np.zeros((N, 93, 93)),
#         'circular_filter_response': np.zeros(N),
#         'mei_masked_images': np.zeros((N, 93, 93)),
#         'mei_masked_response':  np.zeros(N),
#         'circluar_filter_midgrey_images': np.zeros((N, 93, 93)),
#         'circular_filter_midgrey_response': np.zeros(N),
#         'mei_masked_midgrey_images': np.zeros((N, 93, 93)),
#         'mei_masked_midgrey_response': np.zeros(N),

#     }
#%%

with torch.no_grad():
    dataloader = create_dataloader(image_dir, batch_size=1000, crop_size=93, crop_position='center')
    image_size_in_degrees = 2.67  
    for ii, images in enumerate(dataloader):
        for index in tqdm(data_dict.keys()):
            results['full_field_images'] = images.squeeze().cpu().numpy()
            results['full_field_responses'] = model(images.cuda())[:, index].squeeze().cpu().numpy()

            # circular filter
            images_circular = torch.tensor(apply_circular_filter(
                images.squeeze().cpu().numpy(),
                position=data_dict[index]['center_mask_mei'],
                size_in_degrees=image_size_in_degrees,
                return_filter=False
                ))
            results['circular_filter_images'] = images_circular.squeeze().cpu().numpy()
            results['circular_filter_response'] = model(images_circular.cuda().unsqueeze(1))[:, index].squeeze().cpu().numpy()

            # circular filter midgrey
            images_circular_midgrey = torch.tensor(apply_circular_filter(
                images.squeeze().cpu().numpy() - pixel_mean,
                position=data_dict[index]['center_mask_mei'],
                size_in_degrees=image_size_in_degrees,
                return_filter=False
                )) + pixel_mean
            results['circular_filter_midgrey_images'] = images_circular_midgrey.squeeze().cpu().numpy()
            results['circular_filter_midgrey_response'] = model(images_circular_midgrey.cuda().unsqueeze(1))[:, index].squeeze().cpu().numpy()

            # mei masked
            images_mei_masked = images * torch.tensor(data_dict[index]['mask_mei']).unsqueeze(0).unsqueeze(0)
            results['mei_masked_images'] = images_mei_masked.squeeze().cpu().numpy()
            results['mei_masked_response'] = model(images_mei_masked.cuda())[:, index].squeeze().cpu().numpy()

            # mei masked midgrey
            images_mei_masked_midgrey = (images - pixel_mean) * torch.tensor(data_dict[index]['mask_mei']).unsqueeze(0).unsqueeze(0) + pixel_mean
            results['mei_masked_midgrey_images'] = images_mei_masked_midgrey.squeeze().cpu().numpy()
            results['mei_masked_midgrey_response'] = model(images_mei_masked_midgrey.cuda())[:, index].squeeze().cpu().numpy()

            picklesave(f'/project/experiment_data/convnext/results_coen_cagli_index_{index}.pickle', results)


#%%
a = pickleread('/project/experiment_data/convnext/results_coen_cagli_index_0.pickle')
b = pickleread('/project/experiment_data/convnext/results_coen_cagli_index_1.pickle')
# %%
import matplotlib.pyplot as plt
plt.imshow(a['mei_masked_midgrey_images'][100])
plt.show()
plt.imshow(b['mei_masked_midgrey_images'][100])
plt.show()



# %%
