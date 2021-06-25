import logging
'''
import hdf5
import torch
'''
import argparse
'''
import torch
import numpy as np
import os
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
    main()