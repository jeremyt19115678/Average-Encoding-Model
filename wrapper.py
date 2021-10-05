from main import get_ROIs, fetch_image_ids_list
import json
import os

class Wrapper:
    '''
    API/Parameters:
    - model: an instance of a class extended from torch.nn.Module and should support the call to .forward()
    - model_name: a string not containing space that will be used to identify the saved model and its generated plots
    - optim: a torch optimizer (just the class)
    - optim_params: a list of dicts specified by the per-parameter options of torch.optim. Refer to
        https://pytorch.org/docs/stable/optim.html
    - optim_name: a string not containing space that will be used to identify the optimizer
    - dataset_class: the *class* that the dataset should belong to. 
        This *class* should take in two parameters:
            1) a list of image IDs, which are all integers
            2) a single string that specifies the ROI
        and generate an instance of a class extended from torch dataset such that self.model can use this instance
    - dataset_name: should be either "validation", "test", or "train". Used to generate self.dataset
    - loss_func: the loss function should only take in two torch.Tensor, the model prediction and the correct label, and
        return a torch Tensor.
    - roi: a string that specifies the ROI this model is going to be trained for
    '''
    def __init__(self, model, model_name, optim, optim_params, optim_name, dataset_class, dataset_name, loss_func, roi):
        assert isinstance(model_name, str) and isinstance(optim_name, str), "Invalid model_name or optim_name. They must be strings."
        assert dataset_name in ['validation', 'test', 'train'], "Invalid dataset_name. It must be either 'validation', 'test', or 'train'."
        assert roi in get_ROIs(), "{} is not in the list of all available ROIs.".format(roi)
        self.model = model
        model_name = model_name.strip()
        if ' ' in model_name:
            print("model_name contains space character. model_name will be set to all non-space characters up to and not including the first space.")
            model_name = model_name[:model_name.index(' ')]
        self.model_name = model_name
        self.optim = optim(optim_params)
        optim_name = optim_name.strip()
        if ' ' in optim_name:
            print("optim_name contains space character. optim_name will be set to all non-space characters up to and not including the first space.")
            optim_name = optim_name[:optim_name.index(' ')]
        self.optim_name = optim_name
        image_ids = fetch_image_ids_list(dataset_name)
        self.dataset = dataset_class(image_ids, roi)
        self.dataset_name = dataset_name
        self.loss_func = loss_func
        self.roi = roi
        self.num_samples = len(image_ids)