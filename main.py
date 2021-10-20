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
from scipy.special import erf

verbose = True

# print msg if the global variable verbose is True
def log(msg):
    if verbose:
        print(msg)

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

#return a list of images ID in the order of which the images are presented to the subject
def image_sequence():
    # from Zijin's Code
    exp_design_file = os.path.realpath("nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    ordering = exp_design['masterordering'].flatten() - 1
    return ordering.tolist() # cast to Python List

# return the list of IDs of images that are shared across all subjects
# note that some images are not actually shown to all subjects due to experiment constraints
# to get the the IDs of all actually shown shared images, use get_shown_shared_images()
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

# return the list of IDs of the shared images that are actually shown to all subjects
def get_shown_shared_images():
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
    return shown_shared

# return a sorted list of ROIs
def get_ROIs():
    return sorted(['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'L_amygdala', 'L_hippocampus', 'R_amygdala', 'R_hippocampus'])

# returns responses, a list of length 8, wich each element being a nested_dictionary
# nested_dictionary is a dictionary that maps from the number/ID of the image to response_value
# response_value is a dictionary that maps from the ROI name to the list of activation value
def basic_info():
    # brief sanity check
    def sanity_check(responses):
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
        log("Passed all sanity checks.")

    ordering = image_sequence()
    # try to load from file if possible
    if os.path.exists(os.path.realpath("all_subj_roi_activations.json")):
        try:
            with open(os.path.realpath('all_subj_roi_activations.json'), 'r') as f:
                responses = json.load(f)['data']
                # since JSON converts the keys to strings, we have to convert it back
                for nested_dictionary in responses:
                    for key in list(nested_dictionary.keys()):
                        nested_dictionary[int(key)] = nested_dictionary.pop(key)

            sanity_check(responses)
            log("Successfully read from file all_subj_roi_activations.json")
            return responses
        except:
            pass
    # generate from scratch
    log("No suitable all_subj_roi_activations.json found. Generating one.")
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
    sanity_check(responses)

    # if a correct "responses" is generated, save it in JSON Object with one attribute
    # 'data' in file "all_subj_roi_activations.json"
    with open(os.path.realpath('all_subj_roi_activations.json'), 'w') as f:
        obj = {'data': responses}
        f.write(json.dumps(obj))
        log("Successfully created all_subj_roi_activations.json.")
    return responses

# save plot with filename under the folder specified by custom_path
# otherwise, save it under folder "graphs"
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

# set apart the shared data set
# outputs a directory all_images_related_data that contains:
#   shared_images.h5py: a file that contains all 907 shared images that are shown at least once to every subject
#   averagea_activation_ROI.txt: 28 files, one for each ROI, that contains the average activation for all images
def extract_shared_image_set_data():
    responses = basic_info()
    # there should be 907 images that has a "correct answer"
    shown_shared = get_shown_shared_images()
    assert len(shown_shared) == 907
    # read from a h5py file to get the shared images
    image_data_set = h5py.File(os.path.join(os.path.realpath('NSD_stimuli'), 'S1_stimuli_227.h5py'), 'r')
    image_data = np.copy(image_data_set['stimuli']).astype(np.float32) / 255. # convert to [0, 1]
    image_data_set.close()
    assert image_data.shape == (10000, 3, 227, 227)

    shared_image_data = image_data[shown_shared]
    assert (shared_image_data.shape) == (907, 3, 227, 227), "Unexpected shape: {}".format(shared_image_data.shape)

    # save everything in images folder
    all_images_folder = os.path.realpath('all_images_related_data')
    if not os.path.isdir(all_images_folder):
        os.makedirs(all_images_folder)
    # save the shared_image_data somehow (perhaps h5py)
    with h5py.File(os.path.join(all_images_folder, 'shared_images.h5py'), 'w') as f:
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
        with open(os.path.join(all_images_folder, "average_activation_{}.txt".format(roi)), 'w') as f:
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

# checks if a partition dictionary is valid
def functional_partition(partition):
    assert isinstance(partition, dict), "Passed in value is not of type dict"
    right_length = len(partition.keys()) == 3
    right_keys = "validation" in partition and "train" in partition and "test" in partition
    total_IDs = sorted(partition['validation'] + partition['train'] + partition['test']) == [i for i in range(907)]
    return right_length and right_keys and total_IDs

# generates the following JSON (all_images_related_data/partition.json) if it doesn't already exist:
# {
#   'validation': [...], // this will contain the ID's of the validation set
#   'test': [...], // this will contain the ID's of the test set
#   'train': [...] // this will contain the ID's of the train set
# }
def partition_all_images(test_proportion = 0.1, validation_proportion = 0.2, train_proportion = 0.7):
    # we re-index the shared images from 0 to 906
    all_shared_ids = np.arange(907)
    np.random.shuffle(all_shared_ids)
    all_shared_ids = all_shared_ids.astype(int).tolist()
    test_set_count, validation_set_count = int(len(all_shared_ids) * test_proportion), int(len(all_shared_ids) * validation_proportion)
    if os.path.exists(os.path.realpath('all_images_related_data/partition.json')):
        # read in the JSON and report the current partition
        with open(os.path.realpath('all_images_related_data/partition.json'), 'r') as f:
            current_partitions = json.load(f)
        # make sure that the current partition works and has the specified length
        working_partition = functional_partition(current_partitions)
        working_partition = working_partition and len(current_partitions['validation']) == validation_set_count and len(current_partitions['test']) == test_set_count
        # if the current partition works, report it then exit
        if working_partition:
            print("all_images_related_data/partition.json already exists, no updates will be carried out.")
            print("The validation set contains {} images.\nThe training set contains {} images.\nThe testing set contains {} images.".format(len(current_partitions['validation']), len(current_partitions['train']), len(current_partitions['test'])))
            return
    print("There does not currently exist a valid partition at all_images_related_data/partition, will be generating one.")
    #checking parameters
    assert test_proportion >= 0 and validation_proportion >= 0 and train_proportion >= 0, "the proportions of the test, validation, and train sets need to all be positive."
    assert test_proportion + validation_proportion + train_proportion == 1, "the proportions of the test, validation, and train sets need to add up to 1."
    # generating the partitions
    test_set_ids = all_shared_ids[:test_set_count]
    validation_set_ids = all_shared_ids[test_set_count: test_set_count + validation_set_count]
    train_set_ids = all_shared_ids[test_set_count + validation_set_count:]
    # make sure the partitions work
    assert sorted(test_set_ids + validation_set_ids + train_set_ids) == sorted(all_shared_ids)
    # generate needed folders if needed
    if not os.path.isdir(os.path.realpath('all_images_related_data')):
        extract_shared_image_set_data()
    with open(os.path.join("all_images_related_data", "partition.json"), 'w') as f:
        obj = {'validation': validation_set_ids, 'train': train_set_ids, 'test': test_set_ids}
        f.write(json.dumps(obj))
    print("Done.\nThe validation set contains {} images.\nThe training set contains {} images.\nThe testing set contains {} images.".format(len(validation_set_ids), len(train_set_ids), len(test_set_ids)))

# return the list of imageIDs when given a string specifying the dataset
def fetch_image_ids_list(dataset_type: str):
    assert dataset_type in ['test', 'validation', 'train'], "Invalid dataset type: {}".format(dataset_type)
    if os.path.exists(os.path.realpath('all_images_related_data/partition.json')):
        with open(os.path.realpath('all_images_related_data/partition.json'), 'r') as f:
            current_partitions = json.load(f)
            if functional_partition(current_partitions):
                return current_partitions[dataset_type]
            else:
                print("The current all_images_related_data/partition.json is not functional. Call partition_all_images().")
                return
    else:
        print("Unable to fetch image IDs as all_images_related_data/partition.json does not exist. Call partition_all_images().")
        return

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

# evaluate the model (specified by the wrapper) using the specified dataset
# performance evaluated with MSE and Pearson's Correlation
# returns a tuple of two elements, the first is the MSE, the second is the Pearson's Correlation
def evaluate_performance(dataloader, model):
    loss_fn = nn.MSELoss()
    test_loss = []
    predictions = []
    labels = []
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            for i in pred.numpy().astype(np.float32):
                predictions.append(i)
            for i in y.numpy().astype(np.float32):
                labels.append(i)
            test_loss.append(loss_fn(pred, y).item())
    corrcoef_matrix = np.corrcoef(np.array(predictions).squeeze().astype(np.float32), np.array(labels).squeeze().astype(np.float32))
    assert corrcoef_matrix.shape == (2, 2), "{}".format(corrcoef_matrix)
    return np.mean(test_loss), corrcoef_matrix[1,0]

# train the model
# the wrapper should include the optimizing algorithm (and its parameters), the dataset used
# save determines whether the model parameter should be saved after it's trained
# shuffle determines whether the dataset should be shuffled
# batch_size determines the batch size used in optimization
# about_file determines whether a about.json file would be created (to document the experiment)
# additional_info is a dictionary containing all additional information about the experiment (e.g. the beta used in ridge regression)
    # that is to be stored in about.json
def train_model(model_wrapper, epoch, save=False, shuffle=True, batch_size = 64, about_file = True, additional_info = None, experiment_path = None):
    print("Training model\nModel name: {}\nROI: {}\nUsing {} dataset".format(model_wrapper.model_name, model_wrapper.roi, model_wrapper.dataset_name))

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # if there isn't a given experiment_path, the data and graphs generated are stored in `experiments/{model_name}_train_{timestamp}` by default
    if experiment_path == None:
        experiment_path = os.path.realpath('experiments')
        if not os.path.exists(experiment_path):
            os.makedirs(os.path.realpath(experiment_path))
    else:
        if not os.path.isdir(os.path.realpath(experiment_path)):
            print("supplied experiment_path is not a valid experiment folder:\n{}".format(experiment_path))
            return
    
    while True:
        curr_time_ms = str(round(time.time()*1000))
        experiment_folder = os.path.realpath(os.path.join(experiment_path, "{}_train_{}").format(model_wrapper.model_name, curr_time_ms))
        if os.path.exists(experiment_folder): # if an experiment has already been created within the time, we wait for a while
            continue
        else:
            os.makedirs(experiment_folder)
            break # we move on putting the content in the experiment folder knowing that it won't overwrite any other data

    def train(dataloader, model, loss_func, optimizer):
        for batch, (X, y) in enumerate(dataloader):
            if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
                X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # create the data loader
    train_loader = DataLoader(model_wrapper.dataset, shuffle=shuffle, batch_size=batch_size)
    # set up the number of epochs during which the model's performance would be logged
    test_points = np.linspace(0, epoch, num=min(25, epoch), endpoint = False).astype(int)
    train_mse_list, train_r_list = [], []
    # train the model
    for i in range(epoch):
        train(train_loader, model_wrapper.model, model_wrapper.loss_func, model_wrapper.optim)
        # evaluate the model every once in a while and save it
        if i in test_points:
            train_mse,train_r= evaluate_performance(train_loader,model_wrapper.model)
            log("Epoch: {} / {}\tTrain MSE: {}\tTrain r: {}".format(i + 1, epoch, round(train_mse, 4), round(train_r, 4)))
            train_mse_list.append(train_mse)
            train_r_list.append(train_r)
    final_train_error = evaluate_performance(train_loader, model_wrapper.model)
    print("Final Train MSE:: {}\tTrain r:: {}".format(*final_train_error))
    train_mse_list.append(final_train_error[0])
    train_r_list.append(final_train_error[1])

    # save the models if needed
    if save == True:
        model_path = os.path.join(experiment_folder, "{}_model_params_{}.pt".format(model_wrapper.model_name, model_wrapper.roi))
        torch.save(model_wrapper.model.state_dict(), os.path.realpath(model_path))
    
    test_points = np.append(test_points, epoch - 1)
    test_points += 1

    # plot the training curve for MSE and Pearson's Correlation
    plt.plot(test_points, train_mse_list)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Average Training MSE over Time ({})\nTrained with {} using {} dataset".format(model_wrapper.model_name, model_wrapper.optim_name, model_wrapper.dataset_name))
    save_plot("training_mse_curve.png", custom_path=experiment_folder)

    plt.plot(test_points, train_r_list, label="Training Pearson's Correlation")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson's Correlation")
    plt.title("Average Training Correlation over Time ({})\nTrained with {} using {} dataset".format(model_wrapper.model_name, model_wrapper.optim_name, model_wrapper.dataset_name))
    save_plot("training_r_curve.png", custom_path=experiment_folder)

    if about_file:
        about_dict = model_wrapper.about_dict
        about_dict['epoch'] = epoch
        about_dict['batch_size'] = batch_size
        about_dict['mse_list'] = train_mse_list
        about_dict['r_list'] = train_r_list
        if additional_info is not None:
            for key, val in additional_info.items():
                if key not in about_dict:
                    about_dict[key] = val
                else:
                    print('not noting key {} in about.json'.format(key))
        with open(os.path.join(experiment_folder, 'about.json'), 'w') as f:
            f.write(json.dumps(about_dict))
    print("Done.\n")

def setVerbose(val):
    assert isinstance(val, bool)
    global verbose
    verbose = val

def getVerbose():
    global verbose
    print("Verbose = {}".format(verbose))