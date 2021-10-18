import torch
import os
import h5py
import numpy as np
from main import get_ROIs
from torch import nn
from torch.utils.data import Dataset
from alexnet import Alexnet_fmaps
from wrapper import Wrapper

'''
This model takes in a natural number parameter that specifies the power the input would be raised to.
It takes in the flattened input to the of the first fully-connected layer in AlexNet and outputs a
polynomial combination of the inputs raised to the specified power.
'''
class Average_Model_Regression(nn.Module):
    def __init__(self, power: int = 1):
        super(Average_Model_Regression, self).__init__()
        assert isinstance(power, int) and power >= 1, "Power has to be an integer >=1."
        self.power = power
        self.lin = nn.Linear(9216 * power, 1)

    def forward(self, x):
        input_raised_to_power = torch.cat(tuple([torch.pow(x, i) for i in range(1, self.power + 1)]), dim=1)
        return torch.flatten(self.lin(input_raised_to_power))

class Linear_Model_Dataset(Dataset):
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

'''
returns the ridge regression loss given the beta and the model (to get the weights)
'''
def ridge_regression_loss(pred, label, lin_model, beta):
    return torch.mean(torch.pow(pred - label, 2)) + beta * (torch.sum(lin_model.lin.weight) + lin_model.lin.bias)