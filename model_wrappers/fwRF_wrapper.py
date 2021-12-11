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

# adopted from Zijin's code
# modified slightly with help from the paper by Ghislain St-Yves et al.
# "The feature-weighted receptive field: an interpretable encoding model for complex feature spaces"
def make_gaussian_mass(x, y, sigma, n_pix):
    deg = 1.0 # seem to be constant in Zijin's code
    dpix = deg / n_pix
    pix_min = -deg/2. + 0.5 * dpix
    pix_max = deg/2.
    X_mesh, Y_mesh = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix))
    # basically the same as NeuroGen's version, with the only difference being
    # using erf instead of approximating the Gaussian blob integral when
    # sigma >= dpix
    if sigma <= 0:
        Zm = torch.zeros_like(torch.from_numpy(X_mesh))
    else:
        g_mass = np.vectorize(lambda a, b: gaussian_mass(a, b, dpix, dpix, x, y, sigma)) 
        Zm = g_mass(X_mesh, -Y_mesh).astype(np.float32)
    assert tuple(Zm.shape) == (n_pix, n_pix), "Returned matrix is of size {} when feature map side length is {}.".format(tuple(Zm.shape), n_pix)
    return X_mesh, -Y_mesh, Zm

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
    def __init__(self, dataset_type = 'validation', x = 0, y = 0, sigma = 1, input_shape=(1,3,227,227), layer_max_fmaps = 1024, aperture=1.0):
        super(Average_Model_fwRF, self).__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aperture = aperture # I don't really get what this does

        # initialize a tensor of shape (1, 3, 227, 227) of random values from 0-1 (this resembles a picture)
        # we feed the picture into the _fmaps_fn to get an output to get a list of output of each layer
        _x = torch.empty((1,)+input_shape[1:], device=self.device).uniform_(0, 1)
        _fmaps_fn = Alexnet_fmaps()
        all_fmaps = _fmaps_fn(_x)
        _fmaps = all_fmaps[:5] + all_fmaps[6:]
        #self.fmaps_rez = [] # should contain the resolution of the feature maps of each layer
        num_feature_maps = 0
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            #self.fmaps_rez += [_fm.size()[2],]
            if _fm.size()[2] == 1: # this is a fully connected layer
                num_feature_maps += _fm.size()[1] if _fm.size()[1] <= layer_max_fmaps else layer_max_fmaps
            else: #this is a convolutional layer
                num_feature_maps += _fm.size()[1]
        # self.fmaps_rez should contain 27, 27, 13, 13, 13, 1, 1, 1 (refer to README)

        # hyperparameters set by the user
        self.pool_mean_x = x
        self.pool_mean_y = y
        self.pool_variance = sigma

        self.feature_map_weights = torch.rand(num_feature_maps + 1, len(get_ROIs()))

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
        # pool the layers
        pooled_layers = [torch.tensordot(layer, torch.from_numpy(make_gaussian_mass(self.pool_mean_x, self.pool_mean_y, self.pool_variance, layer.shape[2])[2]), dims=([2,3], [0,1])) for layer in output]

        # for all layers with exceeding layer_max_fmaps feature maps, we only keep the layer_max_fmaps ones with the highest variance
        layer_fmap_mask = []
        for layer in pooled_layers:
            if layer.shape[1] <= layer_max_fmaps:
                layer_fmap_mask += [True] * layer.shape[1]
            else:
                variance_list = np.var(layer.cpu().detach().numpy().astype(np.float32), axis=0)
                # get the indices of the maximal variances
                maximal_variances_indices = np.argsort(variance_list)[-layer_max_fmaps:]
                mask = [False] * layer.shape[1]
                for ind in maximal_variances_indices:
                    mask[ind] = True
                layer_fmap_mask += mask
        self.fmap_mask = np.argwhere(np.array(layer_fmap_mask))

    # filter the feature maps until there is only the maximal variance ones remain
    # then pass through the weights
    def forward(self, fmaps):
        #integrals = {}
        with torch.no_grad():
            feature_maps = torch.squeeze(fmaps[:,self.fmap_mask]) # size (batch, features)
            if len(feature_maps.shape) == 1:
                feature_maps = torch.reshape(feature_maps, (1, feature_maps.shape[0]))
            # add bias
            X = torch.cat([feature_maps, torch.ones(feature_maps.shape[0], 1)], dim=1) # size (batch, features + 1)
            return torch.mm(X, self.feature_map_weights)

    def closed_form_solution(self, dataloader, beta):
        with torch.no_grad():
            X_list = [x for batch, (x,y) in enumerate(dataloader)]
            X_without_bias = torch.t(torch.squeeze(torch.cat(X_list)[:, self.fmap_mask])) # size (features, set)
            # have to add the bias term
            X = torch.cat([X_without_bias, torch.ones(1, X_without_bias.shape[1])]) # size (features + 1, set)
            Y_list = [y for batch, (x,y) in enumerate(dataloader)]
            Y = torch.t(torch.cat(Y_list))
            # front half of the closed form solution: (XX^T + beta*I)^-1X
            front_term = torch.mm(torch.linalg.inv(torch.mm(X, torch.t(X)) + beta*torch.eye(len(self.fmap_mask) + 1)), X)
            closed_form_solution = torch.cat([torch.mm(front_term, torch.unsqueeze(yi, 1)) for yi in Y], dim=1)
            self.feature_map_weights = closed_form_solution

class fwRF_Dataset(Dataset):
    # the specific_roi parameter is a misnomer, it's required by the wrapper API, but we wouldn't really use it for
    # specifying the ROI, as we want the Dataset for everything
    # instead, we will let it be a tuple of (x,y,sigma), the hyperparameter for the pooling field
    def __init__(self, partition: list, specific_roi):
        images_path = os.path.realpath('all_images_related_data/shared_images.h5py')
        assert os.path.exists(images_path)
        # get all the images
        image_file = h5py.File(images_path, 'r')
        all_images = np.copy(image_file['image_data']).astype(np.float32)[partition]
        image_file.close()
        # convert images into AlexNet readings and save them
        alexnet = Alexnet_fmaps()
        readings = []
        # all_images should be of shape n, 3, 227, 227
        for image in all_images:
            image_tensor = torch.from_numpy(image.reshape(1, 3, 227, 227))
            fmaps = alexnet(image_tensor)
            readings.append(fmaps[:5] + fmaps[6:]) # skip over the 6th element, which is added for the sake of NN and not previously in fwRF used in neurogen
        all_images_torch = torch.from_numpy(all_images)
        full_output = alexnet(all_images_torch)
        output = full_output[:5] + full_output[6:]
        # pool the layers
        x, y, sigma = specific_roi # again, specific_roi is a misnomer
        pooled_layers = [torch.tensordot(layer, torch.from_numpy(make_gaussian_mass(x, y, sigma, layer.shape[2])[2]), dims=([2,3], [0,1])) for layer in output]
        self.fmaps = torch.cat(pooled_layers, dim=1)

        # note the image ids
        assert isinstance(partition, list) and max(partition) <= 906 and min(partition) >= 0, "Image ID out of range"
        self.image_ids = partition

        # load the label for all ROIs
        rois = get_ROIs()
        roi_activation_map = []
        for roi in rois:
            filename = os.path.realpath('all_images_related_data/average_activation_{}.txt'.format(roi))
            activation_list = np.loadtxt(filename).astype(np.float32)
            assert activation_list.shape == (907, ), "activation_list length is {}, different from expected 907.".format(activation_list.shape)
            relevant_activations = activation_list[partition]
            roi_activation_map.append(relevant_activations)
        self.roi_activation_map = torch.transpose(torch.tensor(roi_activation_map), 0, 1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        input = self.fmaps[index]
        # fetch the label of this image
        label = self.roi_activation_map[index]
        return input, label