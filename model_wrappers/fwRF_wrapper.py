import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from alexnet import Alexnet_fmaps
from scipy.special import erf
from main import get_ROIs, fetch_image_ids_list

def gaussian_mass(x, y, dx, dy, x_mean, y_mean, sigma):
    return 0.25*(erf((x+dx/2-x_mean)/(np.sqrt(2)*sigma)) - erf((x-dx/2-x_mean)/(np.sqrt(2)*sigma))) * (erf((y+dy/2-y_mean)/(np.sqrt(2)*sigma)) - erf((y-dy/2-y_mean)/(np.sqrt(2)*sigma)))

# largely adapted from Zijin's Code in Neurogen
class Average_Model_fwRF(nn.Module):

    '''
    Post-condition: 
    self.aperature is just adapted from code from neurogen, i don't get what this does
    self.fmaps_rez is of type List and its elements (of some numerical type) are the side lengths
                   of the layers in the feature maps
    self.pool_mean_x and self.pool_mean_y is the center of the Gaussian pooling field (type float)
    self.pool_variance is also used to generate the Gaussian pooling field (type float)
    self.feature_map_weights is also a Pytorch tensor with autograd enabled, used in the linear combination of the feature maps (after the pooling field)
    self.bias is the bias that will be added at the end of the linear combination (PyTorch parameter for autograd)
    self.regularization_constant is the constant used in ridge regression for feature_map_weights
    '''
    def __init__(self, dataset_type = 'validation', x = 0, y = 0, sigma = 1, input_shape=(1,3,227,227), fc_layer_max = 1024, aperture=1.0, device=torch.device("cpu")):
        super(Average_Model_fwRF, self).__init__()
        
        self.aperture = aperture # I don't really get what this does

        # initialize a tensor of shape (1, 3, 227, 227) of random values from 0-1 (this resembles a picture)
        # we feed the picture into the _fmaps_fn to get an output to get a list of output of each layer
        _x = torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps_fn = Alexnet_fmaps()
        all_fmaps = _fmaps_fn(_x)
        _fmaps = all_fmaps[:5] + all_fmaps[6:]
        self.fmaps_rez = [] # should contain the resolution of the feature maps of each layer
        num_feature_maps = 0
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
            if _fm.size()[2] == 1: # this is a fully connected layer
                num_feature_maps += _fm.size()[1] if _fm.size()[1] <= fc_layer_max else fc_layer_max
            else: #this is a convolutional layer
                num_feature_maps += _fm.size()[1]
        # self.fmaps_rez should contain 27, 27, 13, 13, 13, 1, 1, 1 (refer to README)

        # should perhaps be random
        self.pool_mean_x = x
        self.pool_mean_y = y
        self.pool_variance = sigma
        self.feature_map_weights = nn.Linear(num_feature_maps, 1)

        # generate the gaussian masses so it's faster to generate
        self.gaussian_masses = [torch.from_numpy(self.make_gaussian_mass(npix)[2]) for npix in self.fmaps_rez]

        # load the validation images into a tensor of shape (n, 3, 227, 227) 
        image_data_set = h5py.File(os.path.realpath('all_images_related_data/shared_images.h5py'), 'r')
        image_data = np.copy(image_data_set['image_data']).astype(np.float32)
        image_data = image_data[fetch_image_ids_list(dataset_type)]
        image_data_set.close()
        assert len(tuple(image_data.shape)) == 4 and image_data.shape[1] == 3 and image_data.shape[2] == image_data.shape[3] == 227, "image_data shape unexpected: {}".format(image_data.shape)
        image_data_torch = torch.from_numpy(image_data)
        # feed the validation images through AlexNet to get the output
        full_output = _fmaps_fn(image_data_torch)
        output = full_output[:5] + full_output[6:]
        # note all the fully connected layers
        fully_connected_layers = []
        for layer_num, layer in enumerate(output):
            if layer.shape[2] == layer.shape[3] == 1:
                fully_connected_layers.append(layer_num)
        self.fc_layer_fmap_mask = {i: [] for i in fully_connected_layers}
        for layer_num in fully_connected_layers:
            if output[layer_num].shape[1] <= fc_layer_max:
                self.fc_layer_fmap_mask[layer_num] = [True for i in range(output[layer_num].shape[1])]
            else:
                image_fc_layer_output = torch.squeeze(output[layer_num]).cpu().detach().numpy().astype(np.float32)
                assert len(image_fc_layer_output.shape) == 2
                variance_list = np.var(image_fc_layer_output, axis=0)
                # get the indices of the maximal variances
                maximal_variances_indices = np.argsort(variance_list)[-fc_layer_max:]
                self.fc_layer_fmap_mask[layer_num] = [True if i in maximal_variances_indices else False for i in range(output[layer_num].shape[1])]

    # adopted from Zijin's code
    # modified slightly with help from the paper by Ghislain St-Yves et al.
    # "The feature-weighted receptive field: an interpretable encoding model for complex feature spaces"
    def make_gaussian_mass(self, n_pix):
        deg = 1.0 # seem to be constant in Zijin's code
        dpix = deg / n_pix
        pix_min = -deg/2. + 0.5 * dpix
        pix_max = deg/2.
        X_mesh, Y_mesh = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix))
        # basically the same as NeuroGen's version, with the only difference being
        # using erf instead of approximating the Gaussian blob integral when
        # sigma >= dpix
        if self.pool_variance<=0:
            Zm = torch.zeros_like(torch.from_numpy(X_mesh))
        else:
            g_mass = np.vectorize(lambda a, b: gaussian_mass(a, b, dpix, dpix, self.pool_mean_x, self.pool_mean_y, self.pool_variance)) 
            Zm = g_mass(X_mesh, -Y_mesh).astype(np.float32)
        assert tuple(Zm.shape) == (n_pix, n_pix), "Returned matrix is of size {} when feature map side length is {}.".format(tuple(Zm.shape), n_pix)
        return X_mesh, -Y_mesh, Zm

    # first calculate the "integrals" of each picture
    # each picture generates a bunch of feature maps in each layer of AlexNet, and these feature maps (basically a matrix)
    # are "dotted" (summed the products of corresponding entries) with the gaussian pooling field (generated using
    # self.pool_variance, self.pool_mean_x, and self.pool_mean_y)
    # each of these feature maps after being dotted would generate a single scalar, which is then weighted by 
    # self.feature_map_weights then summed together (resulting in a linear combination of the feature maps' dot products)
    # this linear combination added with self.bias is the result of a single "forward" pass.
    # TODO: needs fixing, output order has to be consistent with the passed in fmaps
    def forward(self, fmaps):
        integrals = {}
        # for each element in the fmaps (represent the pooling field produced by one layer)
        for layer_num, bad_layer in enumerate(fmaps):
            # the images' feature maps from the layer_num-th of AlexNet
            layer = torch.squeeze(bad_layer, dim=1)
            # for each element in the layer (the feature maps of a single picture)
            for image_num, image_fmaps in enumerate(layer):
                for fmap_num, fmap in enumerate(image_fmaps):
                    assert len(tuple(fmap.shape)) == 2, "fmap.shape = {}".format(fmap.shape)
                    # TODO: if the fmap is from a fully connected layer (shape = (1,1)), and it does not form
                    # the highest variance subset, we don't include it in the integral calculation
                    if tuple(fmap.shape) == (1,1) and not self.fc_layer_fmap_mask[layer_num][fmap_num]:
                        continue
                    # get the "integral" and append it to a list
                    integral = torch.tensordot(fmap, self.gaussian_masses[layer_num])
                    if image_num in integrals:
                        integrals[image_num].append(integral)
                    else:
                        integrals[image_num] = [integral]
        img_keys = sorted(list(integrals.keys()))
        assert img_keys == [i for i in range(len(img_keys))] # because it should be in the same order as they are presented
        integrals_list = [integrals[img_num] for img_num in img_keys]
        integrals_torch = torch.tensor(integrals_list, dtype=torch.float32)
        # get weighted sum of the integrals
        # should be of shape (#num_images)
        return self.feature_map_weights(integrals_torch)

class fwRF_Dataset(Dataset):
    def __init__(self, partition: list, specific_roi: str):
        images_path = os.path.realpath('all_images_related_data/shared_images.h5py')
        assert os.path.exists(images_path)
        # get all the images
        image_file = h5py.File(images_path, 'r')
        all_images = np.copy(image_file['image_data']).astype(np.float32)
        image_file.close()
        # convert images into AlexNet readings and save them
        alexnet = Alexnet_fmaps()
        readings = []
        # all_images should be of shape n, 3, 227, 227
        for image in all_images:
            image_tensor = torch.from_numpy(image.reshape(1, 3, 227, 227))
            fmaps = alexnet(image_tensor)
            readings.append(fmaps[:5] + fmaps[6:]) # skip over the 6th element, which is added for the sake of NN and not previously in fwRF used in neurogen
        self.fmaps = readings
        # note the image ids
        assert isinstance(partition, list) and max(partition) <= 906 and min(partition) >= 0, "Image ID out of range"
        self.image_ids = partition
        # note the roi this fwRF model is for
        assert isinstance(specific_roi, str) and specific_roi in get_ROIs(), "Invalid ROI: {}".format(specific_roi)
        self.roi = specific_roi
        # load in the labels
        filename = os.path.realpath('all_images_related_data/average_activation_{}.txt'.format(self.roi))
        activation_list = np.loadtxt(filename).astype(np.float32)
        assert activation_list.shape == (907, ), "activation_list length is {}, different from expected 907.".format(activation_list.shape)
        self.roi_activation_map = activation_list

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # map from index to the id of the image and the roi
        # get the id of the image
        image_ind = self.image_ids[index]
        input = self.fmaps[image_ind]
        # fetch the label of this image
        label = torch.tensor(self.roi_activation_map[image_ind]).reshape(1)
        return input, label

def ridge_regression_loss(pred, label, model, beta):
    return torch.mean(torch.pow(pred - label, 2)) + beta * (torch.sum(torch.pow(model.feature_map_weights.weight, 2)) + torch.pow(model.feature_map_weights.bias, 2))