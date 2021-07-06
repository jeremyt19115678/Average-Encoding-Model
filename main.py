import logging
import h5py
import argparse
import numpy as np
import os
from scipy.io import loadmat
import re
'''
import torch
import matplotlib.pyplot as plt
'''

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

def sanity_check():
    mismatch_indices = {}
    for a in range(1, 9):
        filepath = os.path.realpath("NSD_stimuli/S{}_stimuli_227.h5py".format(a))
        f1 = h5py.File(filepath, 'r')
        dset1 = f1['stimuli']
        for b in range(a, 9):
            filepath = os.path.realpath("NSD_stimuli/S{}_stimuli_227.h5py".format(b))
            f2 = h5py.File(filepath, 'r')
            dset2 = f2['stimuli'] # should be an array-like structure
            mismatch = []
            for i in range(10000):
                if not np.array_equal(dset1[i], dset2[i]):
                    mismatch.append(i)
            mismatch_indices[(a,b)] = mismatch
    return mismatch_indices
    #print("There are {} mismatches:\n{}".format(len(mismatch_indices), mismatch_indices))

#return a list of the sequence in which the images are presented to the subject
def image_sequence():
    # from Zijin's Code
    exp_design_file = os.path.realpath("nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    ordering = exp_design['masterordering'].flatten() - 1
    return ordering.tolist() # cast to Python List

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
                assert len(res_val) == 28 # there should be 28 ROIs
                for act_list in res_val.values():
                        assert len(act_list) == occurrence[imageID]
                        for activation in act_list:
                            assert isinstance(activation, float)
    print("Passed all sanity checks.")
    return responses

