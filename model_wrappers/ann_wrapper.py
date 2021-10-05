import torch
import os
import h5py
import numpy as np
from main import get_ROIs
from torch import nn
from torch.utils.data import Dataset
from alexnet import Alexnet_fmaps
from wrapper import Wrapper

class Average_Model_NN(nn.Module):
    def __init__(self):
        super(Average_Model_NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return torch.flatten(self.net(x))

# identical to the one in linear_model_wrapper
class ANN_Dataset(Dataset):
    def __init__(self, partition: list, specific_roi: str = None):
        images_path = os.path.realpath('all_images_related_data/shared_images.h5py')
        assert os.path.exists(images_path)
        # get all the images
        image_file = h5py.File(images_path, 'r')
        all_images = np.copy(image_file['image_data']).astype(np.float32)
        image_file.close()
        # convert images into AlexNet readings
        input_tensor = torch.from_numpy(all_images)
        alexnet = Alexnet_fmaps()
        readings = alexnet(input_tensor)[5]
        assert isinstance(readings, torch.Tensor)
        readings = readings.cpu().detach().numpy().astype(np.float32)
        assert readings.shape[1] == 9216
        readings_torch = torch.from_numpy(readings)
        self.fmaps = readings_torch

        assert isinstance(partition, list) and max(partition) <= 906 and min(partition) >= 0, "Image ID out of range"
        self.image_ids = partition
        self.specific_roi = specific_roi
        assert isinstance(self.specific_roi, str) and self.specific_roi in get_ROIs(), "Invalid ROI: {}".format(self.specific_roi)
        
        filename = os.path.realpath('all_images_related_data/average_activation_{}.txt'.format(self.specific_roi))
        activation_list = np.loadtxt(filename).astype(np.float32)
        assert activation_list.shape == (907, ), "activation_list length is {}, different from expected 907.".format(activation_list.shape)
        self.activations = activation_list

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # map from index to the id of the image and the roi
        # get the id of the image and roi
        image_ind = self.image_ids[index]
        input = self.fmaps[image_ind]
        # fetch the label of this image
        label = torch.tensor(self.activations[image_ind])
        return input, label