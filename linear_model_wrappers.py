import torch
import os
import h5py
import numpy as np
from main import get_ROIs
from torch import nn
from torch.utils.data import Dataset
from alexnet import Alexnet_fmaps
from wrapper import Wrapper

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
    # regression_power = 0 means this dataset is NOT for regression model
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
        readings = readings.cpu().detach().numpy()
        assert readings.shape[1] == 9216
        self.fmaps = readings
        '''
        # raise the input to specified power
        nth_power_readings = []
        for reading in readings:
            original_reading = reading
            polynomial_input = original_reading
            for j in range(2, regression_power+1):
                nth_power_reading = np.power(original_reading, j)
                polynomial_input = np.concatenate((polynomial_input, nth_power_reading))
            nth_power_readings.append(polynomial_input)
        self.fmaps = np.array(nth_power_readings)
        '''

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
        input_np = np.array(self.fmaps[image_ind]).astype(np.float32)
        input = torch.from_numpy(input_np)
        # fetch the label of this image
        label = torch.tensor(self.activations[image_ind])
        return input, label