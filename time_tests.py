from baseline.encoding import load_encoding
import torch
import h5py
import os
import numpy as np
from main import get_ROIs, Average_Model_NN
import time
from alexnet import Alexnet_fmaps

# maps roi to the index
# returns None if the subj does not have the roi
def roi_to_index(subj, roi_str):
    all_ROIs = get_ROIs()
    maps = [['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
            ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']]
    for row in maps:
        for roi in row:
            assert roi in all_ROIs
    assert isinstance(subj, int) and 1 <= subj <= 8, "Invalid subject number: {}".format(subj)
    return maps[subj - 1].index(roi_str) if roi_str in maps[subj-1] else None
    
def time_fwRF(specific_roi):
    file = h5py.File(os.path.realpath("validation/shared_images.h5py"), 'r')
    images = np.copy(file['image_data']).astype(np.float32)
    file.close()
    all_activations = []
    ROIs = [specific_roi]
    begin_time = time.time()
    for i in range(1, 9):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier, maps = load_encoding(subject=i, model_name='dnn_fwrf', device=device)
        classifier.eval()
        maps.eval()
        # pixels have to be between 0 and 1 and of dimension 227*227
        subj_activations = {}
        for roi in ROIs:
            subj_activations[roi] = []
        for image in images:
            image_tensor = torch.from_numpy(image.reshape(1, 3, 227, 227))
            pred = classifier(maps(image_tensor)).cpu().detach().numpy()
            for roi in ROIs:
                ind = roi_to_index(i, roi)
                if ind == None:
                    subj_activations[roi].append(np.NAN)
                else:
                    subj_activations[roi].append(pred[ind])
        all_activations.append(subj_activations)
    
    # get the average predictions for each ROI
    avg_predictions = {}
    for roi in ROIs:
        prediction = []
        for i in range(len(images)):
            image_pred = [all_activations[subj][roi][i] for subj in range(8)]
            prediction.append(np.nanmean(image_pred))
        avg_predictions[roi] = prediction
    for preds in avg_predictions.values():
        assert len(preds) == len(images)
    end_time = time.time()
    return end_time - begin_time, avg_predictions

def time_ANN(specific_roi):
    file = h5py.File(os.path.realpath("validation/shared_images.h5py"), 'r')
    images = np.copy(file['image_data']).astype(np.float32)
    file.close()
    ROIs = [specific_roi]
    begin_time = time.time()
    avg_predictions = {}
    alexnet = Alexnet_fmaps()
    for roi in ROIs:
        roi_activations = []
        # load the model
        model = Average_Model_NN()
        model.load_state_dict(torch.load(os.path.abspath('models/NN_model_params_{}.pt'.format(roi))))
        model.eval()
        for image in images:
            image_tensor = torch.from_numpy(image.reshape(1, 3, 227, 227))
            pred = model(alexnet(image_tensor)[5])
            roi_activations.append(pred.cpu().detach().numpy()[0])
        # add entry into avg_predictions
        avg_predictions[roi] = roi_activations
    end_time = time.time()
    return end_time - begin_time, avg_predictions