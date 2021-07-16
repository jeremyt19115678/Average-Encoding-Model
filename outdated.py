"""
IMPORTANT: Collection of outdated functions that were used to examine the data (in a rudimentary way)
These are taken out of main.py and thus might have issue calling some functions.
These are not fully tested. Only here for the sake of keeping records.
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from main import basic_info, save_plot, get_shared_images

# get the spread of the data (when they are shown multiple times to the same subject)
def get_spread(analyze_part="all", log_scale = False):
    if analyze_part != "all" and analyze_part != "shared":
        print("analyze_part parameter must either be 'all' or 'shared'.")
        return
    print("Reading from .txt files...")
    responses = basic_info()
    if analyze_part == "all":
        lengths = [10000, 10000, 6234, 5445, 10000, 6234, 10000, 5445]
        for subject in range(8):
            print("Plotting for subject {}...".format(subject + 1)) # 0-index to 1-index
            nested_dict = responses[subject]
            spreads = []
            for activation_list in nested_dict.values():
                assert len(activation_list) == 28
                for activations in activation_list.values():
                    try:
                        assert len(activations) == 3
                        diff = max(activations) - min(activations)
                        spreads.append(diff)
                    except:
                        break
            assert len(spreads) == lengths[subject] * 28, "Spreads length {} is less than expected {}.".format(len(spreads), lengths[subject] * 28)
            plt.xlabel("Difference in Activation")
            plt.ylabel("Count")
            if not log_scale:
                bins = np.linspace(0, 2.5, num=21)
            else:
                bins = np.logspace(np.log10(0.02), np.log10(2), 21)
                plt.gca().set_xscale("log")
            plt.hist(spreads, bins=bins)
            if not log_scale:
                save_plot("subj0{}_spread.png".format(subject + 1)) # convert from 0-index to 1-index
            else:
                save_plot("subj0{}_spread_log.png".format(subject + 1)) # convert from 0-index to 1-index
    elif analyze_part == "shared":
        shared_images = get_shared_images()
        assert len(set(shared_images)) == 1000 and len(shared_images) == 1000, "Shared images are not unique/shared images number more than 1000"
        shared_images_all_three = []
        for image in shared_images:
            try:
                for i in range(8):
                    activation_dict = responses[i][image]
                    for values in activation_dict.values():
                        assert len(values) == 3
                shared_images_all_three.append(image)
            except:
                continue
        # there should be 515 images that are shown all 3 times to every subject
        assert len(shared_images_all_three) == 515
        for subject in range(8):
            spreads = []
            for image in shared_images_all_three:
                for activation_list in responses[subject][image].values():
                    diff = max(activation_list) - min(activation_list)
                    spreads.append(diff)
            assert len(spreads) == 515 * 28
            plt.xlabel("Difference in Activation")
            plt.ylabel("Count")
            if not log_scale:
                bins = np.linspace(0, 2.5, num=21)
            else:
                bins = np.logspace(np.log10(0.02), np.log10(2), 21)
                plt.gca().set_xscale("log")
            plt.hist(spreads, bins=bins)
            if not log_scale:
                save_plot("subj0{}_shared_set_spread.png".format(subject + 1)) # convert from 0-index to 1-index
            else:
                save_plot("subj0{}_shared_set_spread_log.png".format(subject + 1)) # convert from 0-index to 1-index
        print("Passed Assertion.")

# visualize the shared 1000 stimuli
def visualize_shared_stimuli():
    get_spread(analyze_part="shared")