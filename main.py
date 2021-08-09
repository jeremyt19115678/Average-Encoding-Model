import logging
import h5py
import argparse
import numpy as np
import os
from scipy.io import loadmat
import re
import matplotlib.pyplot as plt
import time
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from alexnet import Alexnet_fmaps

# return the arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new model from scratch. If this flag is not set, the script tries to load a pre-trained model.")
    parser.add_argument("--epoch", action="store", default=1000, type=int, help="Epoch to train the model for. Default value: 1000")
    parser.add_argument("--lr", action = "store", default=0.0001, type=float, help="The learning rate used in training. Default value: 0.0001")
    parser.add_argument("--img", action="store", help="The filename of the image to be evaluated.")
    parser.add_argument("--roi", action="store", type=int, default=0, help="ROI ID, range=[0, 24]. Default value: 0.")
    parser.add_argument("--logs",action="store_true", help="Does logging in avg_encoding_model.log. Use for debugging.")
    #parser.add_argument("--optimizer", action="store", default='adam', help="The optimizer used to train the model. Default value: 'sgd'")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.logs == True:
        logging.basicConfig(filename="avg_encoding_model.log", filemode='w+', level=logging.INFO)
    if args.train == False:
        print("We will try to load a model. If we succeed, we proceed accordingly. If we fail, we output error message.")
        if args.img == None:
            print("Specify a file to be evaluated by the model.")
            return
        if args.roi < 0 or args.roi > 24: #out of the acceptable range
            print("The ROI we are trying to evaluate is out of the acceptable range: [0, 24].")
            return
        print("If we successfully loaded a model, we will try to load the image from: {}".format(args.img))
        print("If we successfully loaded the image, we will output the activation for ROI {}".format(args.roi))
    else:
        print("We will try to train a new model from scratch then save it.")
        print("Relevant training parameters:\n\tmax epoch:: {}\n\tlr:: {}".format(args.epoch, args.lr))

if __name__ == "__main__":
    '''
    for i in range(1, 9):
        filepath = os.path.realpath("NSD_stimuli/S{}_stimuli_227.h5py".format(i))
        f = h5py.File(filepath, 'r')
        dset = f['stimuli']
        print(dset.shape)
    '''
    pass

#return a list of the sequence in which the images are presented to the subject
def image_sequence():
    # from Zijin's Code
    exp_design_file = os.path.realpath("nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    ordering = exp_design['masterordering'].flatten() - 1
    return ordering.tolist() # cast to Python List

# return the list of IDs of images that are shared across all subjects
def get_shared_images():
    exp_design_file = os.path.realpath("nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    image_ID_mapping = exp_design['subjectim'] - 1
    assert image_ID_mapping.shape == (8, 10000)
    image_ID_mapping = image_ID_mapping.transpose().tolist()
    shared_list = []
    for image_num, ids in enumerate(image_ID_mapping):
        if len(set(ids)) == 1: # only one element in the ids
            shared_list.append(image_num)
    return shared_list

# return a sorted list of ROIs
def get_ROIs():
    return sorted(['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'L_amygdala', 'L_hippocampus', 'R_amygdala', 'R_hippocampus'])

# get some basic info regarding the data set:
# if an image is shown multiple times to a subject, how different are the activations each time?
# visualize the shared 1000 images activation 
def basic_info():
    #get the sequence in which the images are presented to the subjects
    # number of images presented to each subject
    # e.g. the number of images shown to subject 3 is at the 3rd element in this list
    ordering = image_sequence()
    # responses is a list of length 8, wich each element being a nested_dictionary
    # nested_dictionary is a dictionary that maps from the number/ID of the image to response_value
    # response_value is a dictionary that maps from the ROI name to the list of activation value
    responses = []
    for i in range(1,9):
        nested_dictionary = {}
        directory_str = os.path.realpath("roiavgbeta/subj0{}".format(i))
        directory = os.fsencode(directory_str)
        activations, rois = [], []
        for file in os.listdir(directory):
            filename = os.path.join(directory_str, os.fsdecode(file))
            roi = re.search("meanbeta_(.*).txt", filename).group(1)
            activation = np.genfromtxt(filename).tolist()
            rois.append(roi)
            activations.append(activation)
        # activations is now a 2D matrix, with the n-th element being the list of activations in the 
        # nth roi in rois (a simple array of strings)
        for ind in range(len(activations[0])):
            image_ID = ordering[ind]
            response_value = nested_dictionary.get(image_ID, {})
            for roi_ind, roi in enumerate(rois):
                activation_list = response_value.get(roi, [])
                activation_list.append(activations[roi_ind][ind])
                response_value[roi] = activation_list
            nested_dictionary[image_ID] = response_value
        responses.append(nested_dictionary)
    
    # some sanity checks
    num_trials = [30000, 30000, 24000, 22500, 30000, 24000, 30000, 22500]
    num_distinct = [10000, 10000, 9411, 9209, 10000, 9411, 10000, 9209]
    assert len(responses) == 8 # there should be 8 subjects
    for i in range(8):
        assert len(responses[i].keys()) == num_distinct[i] # number of distinct images of the subject should match
        occurrence = [0 for i in range(10000)] #number of times each image was shown to the subject
        for j in ordering[:num_trials[i]]:
            occurrence[j] += 1
        assert sum(occurrence) == num_trials[i] # number of trials should match
        for imageID, res_val in responses[i].items():
                assert sorted(list(res_val.keys())) == get_ROIs(), "Expected: {}\n, Got: {}\n".format(get_ROIs(), sorted(list(res_val.keys()))) # check if the ROIs are expected
                for act_list in res_val.values():
                        assert len(act_list) == occurrence[imageID]
                        for activation in act_list:
                            assert isinstance(activation, float)
    print("Passed all sanity checks.")
    return responses

# save plot with filename
def save_plot(filename, custom_path = None):
    if custom_path == None:
        graphics_folder = os.path.realpath('graphs')
        if not os.path.isdir(graphics_folder):
            os.makedirs(graphics_folder)
    else:
        graphics_folder = os.path.realpath(custom_path)
        assert os.path.isdir(custom_path), "{} is not a valid path.".format(custom_path)
    plt.savefig(os.path.join(graphics_folder, filename))
    plt.clf()

# set apart the validation data set
# output a file that contains all 907 shared images that are shown at least once to every subject
# a file for each ROI for the average activation (across all trials)
def extract_validation_set():
    responses = basic_info()
    shared_set = get_shared_images() # will contain the list of shared images
    # there should be 907 images that has a "correct answer"
    shown_shared = []
    for img in shared_set:
        in_all = True
        for nested_dict in responses:
            if img not in nested_dict:
                in_all = False
                break
        if in_all:
            shown_shared.append(img)
    assert len(shown_shared) == 907
    # read from a h5py file to get the shared images
    image_data_set = h5py.File(os.path.join(os.path.realpath('NSD_stimuli'), 'S1_stimuli_227.h5py'), 'r')
    image_data = np.copy(image_data_set['stimuli']).astype(np.float32) / 255. # convert to [0, 1]
    image_data_set.close()
    assert image_data.shape == (10000, 3, 227, 227)

    shared_image_data = image_data[shown_shared]
    assert (shared_image_data.shape) == (907, 3, 227, 227), "Unexpected shape: {}".format(shared_image_data.shape)

    # save everything in validation set
    validation_folder = os.path.realpath('validation')
    if not os.path.isdir(validation_folder):
        os.makedirs(validation_folder)
    # save the shared_image_data somehow (perhaps h5py)
    with h5py.File(os.path.join(validation_folder, 'shared_images.h5py'), 'w') as f:
        f.create_dataset("image_data", data=shared_image_data)
    # save the stimuli per ROI
    roi_activation = {} # maps ROI (28 of them) to list of activations (907 of them)
    # 907 average activations, ignoring NaN (because some ROIs are not present in all subjects)
    ROIs = get_ROIs()
    for roi in ROIs:
        img_activations = []
        for img in shown_shared:
            subject_responses = [np.mean(responses[subj][img][roi]) for subj in range(8)]
            img_activations.append(subject_responses)
        assert len(img_activations) == 907
        roi_activation[roi] = img_activations
    assert len(roi_activation.keys()) == 28, "Number of activation ({}) is different from expected 28.".format(len(roi_activation.keys()))
    for roi, activations in roi_activation.items():
        with open(os.path.join(validation_folder, "average_activation_{}.txt".format(roi)), 'w') as f:
            texts = ""
            nan_indices = []
            for image_activations in activations:
                texts += str(np.nanmean(image_activations)) + '\n'
                nan_index_line = np.where(np.isnan(np.array(image_activations)))[0].tolist()
                nan_index_line = tuple(sorted(list(set(nan_index_line))))
                nan_indices.append(nan_index_line)
            f.write(texts[:-1]) # get rid of the last '\n
            if len(set(nan_indices)) == 1 and len(nan_indices[0]) > 0:
                one_indexed_nan_indices = [i+1 for i in nan_indices[0]]
                print("Subjects that do not have ROI {}: {}".format(roi, one_indexed_nan_indices))
            if len(set(nan_indices)) != 1:
                print("Unexpected NaN value encountered in ROI: {}".format(roi))

# generate a folder for a certain experiment, identified by the time stamp (and perhaps setup)
def generate_k_fold_dataset(partition: int):
    # make sure validation set exists and is in normal state
    validation_folder = os.path.realpath('validation')
    if not os.path.isdir(validation_folder):
        extract_validation_set()
    rois = []
    for file in os.listdir(validation_folder):
        filename = os.path.join(validation_folder, os.fsdecode(file))
        file_extension = os.path.splitext(filename)
        if file_extension[1] == '.txt':
            roi = re.search("average_activation_(.*).txt", filename).group(1)
            rois.append(roi)
    if sorted(rois) != get_ROIs():
        extract_validation_set()
    print("Validation folder should exist and is functional.")
    
    # make the experiment folder if it doesn't exist
    experiment_path = os.path.realpath('experiments')
    if not os.path.exists(experiment_path):
        os.makedirs(os.path.realpath(experiment_path))
    while True:
        curr_time_ms = str(round(time.time()*1000))
        experiment_folder = os.path.realpath(os.path.join(experiment_path, "cross_validation_{}").format(curr_time_ms))
        if os.path.exists(experiment_folder): # if an experiment has already been created within the time, we wait for a while
            continue
        else:
            os.makedirs(experiment_folder)
            break # we move on putting the content in the experiment folder knowing that it won't overwrite any other data
    # load the images to get the length of the validation images
    images = h5py.File(os.path.join(validation_folder, 'shared_images.h5py'), 'r')
    length = len(np.copy(images['image_data'])) if 'image_data' in images else None
    images.close()
    assert length != None and length >= 0, "Read from shared file should've been effective"
    indices = np.arange(length)
    np.random.shuffle(indices)
    indices_groups = np.array_split(indices, partition) #should be a python list of np arrays
    partitions = [{'test': indices_groups[i].astype(int).tolist(), 'train': np.concatenate(tuple(indices_groups[:i]) + tuple(indices_groups[i+1:])).astype(int).tolist()} for i in range(partition)]
    # more sanity checks
    for dict in partitions:
        used_indices = []
        for li in dict.values():
            used_indices += li
        used_indices_sorted = sorted(used_indices)
        assert used_indices_sorted == [i for i in range(length)], "{}".format(used_indices_sorted.shape) #"Did not use all of validation set for testing/training"
    with open(os.path.join(experiment_folder, 'about.json'), 'w') as f:
        obj = {'partitions': partitions, 'completed': False}
        f.write(json.dumps(obj))
    print("Experiment file ({}) should be created.".format(experiment_folder))

# returns a np array of shape (length,) and 0 at every index except position (where there's a 1)
def one_hot_np(length: int, position: int):
    assert isinstance(position, int) and 0 <= position <= length
    arr = np.zeros(length).tolist()
    arr[position] = 1
    return np.array(arr)

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

class Average_Model_Regression(nn.Module):
    def __init__(self, power: int = 1):
        super(Average_Model_Regression, self).__init__()
        assert isinstance(power, int) and power >= 1, "Power has to be an integer >=1."
        self.lin = nn.Linear(9216 * power, 1)

    def forward(self, x):
        return torch.flatten(self.lin(x))

# largely adapted from Zijin's Code in Neurogen
class Average_Model_fwRF(nn.Module):

    def __init__(self, _fmaps_fn, _nonlinearity=None, input_shape=(1,3,227,227), aperture=1.0, device=torch.device("cpu")):
        super(Average_Model_fwRF, self).__init__()
        
        self.aperture = aperture

        # initialize a tensor of shape (1, 3, 227, 227) of random values from 0-1 (this resembles a picture)
        # we feed the picture into the _fmaps_fn to get an output to get a list of output of each layer
        _x = torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = [] # should contain the resolution of the feature maps of each layer
        num_feature_maps = 0
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
            num_feature_maps += _fm.size()[1]
        # self.fmaps_rez should contain 27, 27, 13, 13, 13, 1, 1, 1 (refer to README)

        # should perhaps be random
        self.pool_mean_x = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.pool_mean_y = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.pool_variance = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        self.feature_map_weights = nn.Linear(num_feature_maps, 1)
 
    # adopted from Zijin's code
    # modified slightly with help from the paper by Ghislain St-Yves et al.
    # "The feature-weighted receptive field: an interpretable encoding model for complex feature spaces"
    def make_gaussian_mass(self, n_pix):
        deg = 1.0 # seem to be constant in Zijin's code
        dpix = deg / n_pix
        pix_min = -deg/2. + 0.5 * dpix
        pix_max = deg/2.
        X_mesh, Y_mesh = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix))
        Xm = torch.from_numpy(X_mesh, dtype=torch.float32)
        Ym = torch.from_numpy(Y_mesh, dtype=torch.float32)
        # very different from NeuroGen's version, copied from paper
        if self.pool_variance<=0:
            Zm = torch.zeros_like(torch.from_numpy(Xm))
        else:
            Zm = 1. / torch.sqrt((2*self.pool_variance**2)*np.pi) * torch.exp(-((Xm-self.pool_mean_x)**2 + (-Ym-self.pool_mean_y)**2) / (2*self.pool_variance**2))
        assert tuple(Zm.shape) == (n_pix, n_pix), "Returned matrix is of size {} when feature map side length is {}.".format(tuple(Zm.shape), n_pix)
        return Zm

    def forward(self, fmaps):
        assert len(tuple(fmaps.shape)) == 4 and fmaps.shape[0] == 1, "Need to implement how to deal with batch of images"
        fmaps = torch.squeeze(fmaps)
        assert len(tuple(fmaps.shape)) == 3, "fmaps is of {}-D instead of 3-D".format(len(tuple(fmaps.shape)))
        integrals = []
        # for each element in the fmaps (represent the pooling field produced by one layer)
        for layer_num, layer in enumerate(fmaps):
            # generate the pooling field
            pooling_field = self.make_gaussian_mass(self.fmaps_rez[layer_num])
            # for each element in the layer (a single feature map)
            for fmap in layer:
                assert len(tuple(fmap.size)) == 2
                # get the "integral" and append it to a list
                integral = torch.tensordot(fmap, pooling_field)
                integrals.append(integral)
        # get weighted sum of the integrals
        integrals_tensor = torch.tensor(integrals, dtype=torch.float32)
        return self.feature_map_weights(integrals_tensor)

class Custom_Dataset(Dataset):
    # regression_power = 0 means this dataset is NOT for regression model
    def __init__(self, partition: list, specific_roi: str = None, regression_power: int = 0):
        assert isinstance(regression_power, int) and regression_power >= 0, "regression_power has to be greater than or equal to 0"
        images_path = os.path.realpath('validation/shared_images.h5py')
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
        # if this is not a regression dataset
        if regression_power == 0:
            self.fmaps = readings
        else: #this is a regression dataset
            nth_power_readings = []
            for reading in readings:
                original_reading = reading
                polynomial_input = original_reading
                for j in range(2, regression_power+1):
                    nth_power_reading = np.power(original_reading, j)
                    polynomial_input = np.concatenate((polynomial_input, nth_power_reading))
                nth_power_readings.append(polynomial_input)
            self.fmaps = np.array(nth_power_readings)

        # generate one hot from partition, which should be a list of numbers
        assert isinstance(partition, list) and max(partition) <= 906 and min(partition) >= 0, "Image ID out of range"
        self.image_ids = partition
        self.specific_roi = specific_roi
        if self.specific_roi == None:
            self.rois = get_ROIs()
        else:
            assert isinstance(self.specific_roi, str) and self.specific_roi in get_ROIs(), "Invalid ROI: {}".format(self.specific_roi)
            self.rois = [self.specific_roi]
        self.roi_activation_map = {}
        for roi in self.rois:
            filename = os.path.realpath('validation/average_activation_{}.txt'.format(roi))
            activation_list = np.loadtxt(filename).astype(np.float32)
            assert activation_list.shape == (907, ), "activation_list length is {}, different from expected 907.".format(activation_list.shape)
            self.roi_activation_map[roi] = activation_list

    def __len__(self):
        return len(self.image_ids) * len(self.rois)

    def __getitem__(self, index):
        # map from index to the id of the image and the roi
        # get the id of the image and roi
        image_ind = self.image_ids[int(index / len(self.rois))]
        roi_id = index % len(self.rois)
        roi = self.rois[roi_id]
        # concatenate the image feature w/ the one hot encoding of roi for the input Tensor
        if self.specific_roi == None:
            input_np = np.concatenate((self.fmaps[image_ind], one_hot_np(len(self.rois), roi_id))).astype(np.float32)
        else:
            input_np = np.array(self.fmaps[image_ind]).astype(np.float32)
        input = torch.from_numpy(input_np)
        # fetch the label of this image
        label = torch.tensor(self.roi_activation_map[roi][image_ind])
        return input, label

# find the newest cv folder, run the CV on the model described in description
# get the MSE and save description
# regression_power parameter is by default 0, which indicates a NN is being tested
# if it's any other positive integer, it is a linear regression model
def run_k_fold_cv(optimizer_type = "Adam", learning_rate = 0.00002, epoch = 200, loss_type = "MSE", roi = None, verbose=True, regression_power = 0):
    description = "This is a CV on a linear regression model with 9216 * n + 1 input dimension, " + \
                "where n, is the power to which the independent variables are raised." + \
                "The 9216 of the input comes from the " + \
                "output of the last convolutional layer of AlexNet and the 1 is the bias term."

    # assert that roi is okay
    if roi != None:
        assert isinstance(roi, str) and roi in get_ROIs(), "Invalid roi: {}".format(roi)

    # assert that the regression_power is okay
    assert isinstance(regression_power, int) and regression_power >= 0, "Invalid regression_power: has to be integer >= 0."

    # find the oldest CV folder
    directory_str = os.path.realpath('experiments')
    directory = os.fsencode(directory_str)
    timestamps = []
    for file in os.listdir(directory):
        folder = os.path.join(directory_str, os.fsdecode(file))
        if "cross_validation_" in folder:
            # open the json if it exists, check if it's completed
            about_path = os.path.join(folder, 'about.json')
            if os.path.exists(about_path):
                with open(about_path, 'r') as f:
                    cv_info = json.load(f)
                    if isinstance(cv_info, dict) and 'partitions' in cv_info and 'completed' in cv_info and not cv_info['completed']:
                        timestamps.append(float(re.search("cross_validation_(.*)", folder).group(1)))
    if len(timestamps) == 0:
        print("Cannot find any cross validation that haven't been done already. Consider calling generate_k_fold_dataset()")
        return
    filename = os.path.join(directory_str, "cross_validation_{}".format(str(int(min(timestamps)))))
    print("Running experiment dataset at: {}".format(filename))

    # generate the Torch Dataset and the DataLoader
    with open(os.path.join(filename, "about.json"), 'r') as f:
        cv_info = json.load(f)

    num_partitions = len(cv_info['partitions'])
    print("Running {}-fold CV.\nEpoch: {}\nOptimizer: {}\nLR: {}".format(num_partitions, epoch, optimizer_type, learning_rate))
    print("ROI: {}".format(roi if roi != None else "all"))
    print("Model Type: {}".format("NN" if regression_power == 0 else "Regression (power = {})".format(regression_power)))

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # using -r as a loss function
    # most likely not going to be used, as it blows up the weights
    def r_loss(prediction, label):
        def cov(x, y):
            assert x.size() == y.size() and len(tuple(x.size())) == len(tuple(y.size())) == 1, "Size {} and {} either mismatch or is not 1D.".format(x.size(), y.size())
            n = x.size()[0]
            x_mean, y_mean = torch.mean(x), torch.mean(y)
            return torch.dot(x - x_mean, y - y_mean) / (n - 1)
        return (-cov(prediction, label)) / torch.sqrt(cov(prediction, prediction)).item() / torch.sqrt(cov(label, label)).item()

    # functions to be called during training and testing
    def train(dataloader, model, loss_type, optimizer):
        if loss_type == "MSE":
            loss_fn = nn.MSELoss()
        elif loss_type == "r":
            loss_fn = r_loss
        for batch, (X, y) in enumerate(dataloader):
            if loss_type == "r" and tuple(y.shape)[0] <= 1:
                print("Skipping batch of X, y of size {}, {}.".format(X.shape, y.shape))
                continue
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(dataloader, model):
        loss_fn = nn.MSELoss()
        test_loss = []
        predictions = []
        labels = []
        with torch.no_grad():
            for X,y in dataloader:
                pred = model(X)
                for i in pred.numpy().astype(np.float):
                    predictions.append(i)
                for i in y.numpy().astype(np.float):
                    labels.append(i)
                test_loss.append(loss_fn(pred, y).item())
        corrcoef_matrix = np.corrcoef(np.array(predictions).astype(np.float32), np.array(labels).astype(np.float32))
        assert corrcoef_matrix.shape == (2, 2), "{}".format(corrcoef_matrix)
        return np.mean(test_loss), corrcoef_matrix[1,0]

    error_list = []
    # CV itself
    for partition_num, partition in enumerate(cv_info['partitions']):
        print("Training fold {}".format(partition_num + 1))
        test_set = partition['test']
        train_set = partition['train']
        test_set_torch = Custom_Dataset(test_set, specific_roi=roi, regression_power=regression_power)
        train_set_torch = Custom_Dataset(train_set, specific_roi=roi, regression_power=regression_power)

        # create the loader
        test_loader = DataLoader(test_set_torch)
        train_loader = DataLoader(train_set_torch, shuffle=True, batch_size = 64)

        # train the model (from PyTorch tutorial)
        if regression_power == 0: # this indicates we're training an NN
            model = Average_Model_NN()
        else:
            model = Average_Model_Regression(regression_power)
        if not loss_type == "MSE" and not loss_type == "r":
            print("Unknown type of loss. Please try either 'MSE' or 'r'.")
            return
        if optimizer_type == "SGD":
            optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
        if optimizer_type == "Adam":
            optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        test_points = np.linspace(0, epoch, num=min(25, epoch), endpoint = False).astype(int)
        mse_dict = {}
        for i in range(epoch):
            train(train_loader, model, loss_type, optim)
            # evaluate the model every once in a while and save it
            if i in test_points:
                test_error, test_r = test(test_loader, model)
                train_error,train_r= test(train_loader,model)
                if verbose:
                    print("Epoch: {} / {}\tTrain MSE: {}\tTrain r: {}\tTest MSE: {}\tTest r: {}".format(i + 1, epoch, round(train_error, 4), round(train_r, 4), round(test_error, 4), round(test_r, 4)))
                new_test_list = mse_dict.get('test', [])
                new_test_list.append(test_error)
                new_test_r_list = mse_dict.get('test_r', [])
                new_test_r_list.append(test_r)
                new_train_list = mse_dict.get('train', [])
                new_train_list.append(train_error)
                new_train_r_list = mse_dict.get('train_r', [])
                new_train_r_list.append(train_r)
                mse_dict['test'] = new_test_list
                mse_dict['train']= new_train_list
                mse_dict['test_r']=new_test_r_list
                mse_dict['train_r']=new_train_r_list
        final_test_error, final_train_error = test(test_loader, model), test(train_loader, model)
        print("Final Test MSE:: {}\tTest r:: {}".format(*final_test_error))
        print("Final Train MSE:: {}\tTrain r:: {}".format(*final_train_error))
        mse_dict['test'].append(final_test_error[0])
        mse_dict['train'].append(final_train_error[0])
        mse_dict['test_r'].append(final_test_error[1])
        mse_dict['train_r'].append(final_train_error[1])
        error_list.append(mse_dict)

    # write back to about.json
    with open(os.path.join(filename, "about.json"), 'w') as f:
        cv_info['completed'] = True
        cv_info['description'] = description
        cv_info['errors'] = error_list
        f.write(json.dumps(cv_info))
    print("Updated about.json.")

    # plot the progress from error_list
    all_folds_train_error = np.array([i['train'] for i in error_list])
    all_folds_test_error  = np.array([i['test'] for i in error_list])
    all_folds_train_r = np.array([i['train_r'] for i in error_list])
    all_folds_test_r  = np.array([i['test_r'] for i in error_list])
    avg_train_error_list = [np.mean(all_folds_train_error[:,i]) for i in range(all_folds_train_error.shape[1])]
    avg_test_error_list  = [np.mean(all_folds_test_error [:,i]) for i in range(all_folds_test_error.shape [1])]
    avg_train_r_list = [np.mean(all_folds_train_r[:,i]) for i in range(all_folds_train_r.shape[1])]
    avg_test_r_list  = [np.mean(all_folds_test_r [:,i]) for i in range(all_folds_test_r.shape [1])]

    test_points = np.append(test_points, epoch - 1)
    test_points += 1
    plt.plot(test_points, avg_train_error_list, label="Avg. Training MSE")
    plt.plot(test_points, avg_test_error_list, label ="Avg. Testing MSE")
    # plot the models' progression over time for all folds
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    if roi == None:
        plt.title("Average Training/Testing MSE over Time\n({}, lr={}, loss={})".format(optimizer_type, learning_rate, loss_type))
    else:
        plt.title("Average Training/Testing MSE over Time\n({}, lr={}, loss={}, roi={})".format(optimizer_type, learning_rate, loss_type, roi))
    plt.legend()
    save_plot("avg_mse_curve.png", custom_path=filename)

    plt.plot(test_points, avg_train_r_list, label="Avg. Training r")
    plt.plot(test_points, avg_test_r_list, label ="Avg. Testing r")
    # plot the models' progression over time for all folds
    plt.xlabel("Epoch")
    plt.ylabel("Pearson's Correlation")
    if roi == None:
        plt.title("Average Pearson's Correlation over Time\n({}, lr={}, loss={})".format(optimizer_type, learning_rate, loss_type))
    else:
        plt.title("Average Pearson's Correlation over Time\n({}, lr={}, loss={}, roi={})".format(optimizer_type, learning_rate, loss_type, roi))
    plt.legend()
    save_plot("avg_r_curve.png", custom_path=filename)