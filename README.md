# About
This is a project written by Jeremy Tsai (UCLA CS '23) during the Cornell NeuroNex 2021 Program under the instruction/mentorship of Professor Mert Subuncu and Zijin Gu. The project's goal is to build a group-level encoding model that predicts how a particular image will activate the brain on average for a given Region of Interest (ROI). The model would be trained using the NSD data.  
# Setting Up
Include conda virtual environment information here.
The following is essentially identical to the `conda` documentation (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
1. Create the virtual environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
    The first line of the `environment.yml` file sets the new environment's name. It is currently set to `avg-model`.  

2. Activate the new environment through: `conda activate myenv`. If `environment.yml` has not been tampered with, `myenv` should be `avg-model`.  

3. Verify that the new enviroment was installed correctly:
    ```bash
    conda env list
    ```
    You should see something like:
    ```bash
    myenv             *  /opt/anaconda3/envs/avg-model
    ```
    Again, if `environment.yml` has not been tampered with, `myenv` should be `avg-model`.
# Acknowledgements
This project is heavily derived from the work of my mentor, Zijin Gu. Refer to her original work (NeuroGen) here: https://github.com/zijin-gu/NeuroGen. Here is a link to the paper documenting her work: https://arxiv.org/abs/2105.07140#:~:text=NeuroGen%20combines%20an%20fMRI%2Dtrained,of%20macro%2Dscale%20brain%20activation.
# Training Data
Part of the training data are listed in Zijin's aforementioned GitHub NeuroGen repository, under the file `roiavgbeta`. These are the activation (a single float number) of the particular ROI of the particular subject. The image that causes this activation is too big to be uploaded to GitHub. Contact Jeremy for this data. Note that even with these raw `.h5py` files are not sorted in the order they are presented to the subjects. These sorting information are stored in the `nsd_expdesign.mat` file, which is also omitted from this GitHub repository because of concerns that NSD data has not been made publicly available yet.  
There is a relatively small number of images that are shown to all 8 subjects of the NSD data set. These shared images can be easily used to train the data, because there is a clear correct label (i.e. the average brain activation of all 8 subjects to the specific shared image). There are also images that are only shown to a few subjects, but not all 8 of them. We call these images the partially-shared images. The way we are using to utilize these data is: **STILL UNDECIDED**. Similarly, images that are only shown to one individual (referred to as the _unique pictures_) are also utilized in an **UNDECIDED** way.  
Note that the `validation` folder holds the validation/test set (**the precise use of data in this folder is not yet decided yet, so `validation` might turn out to be a misnomer**). There is also a `shared_images.h5py` that you might notice in the scripts, but it is not included in the script because once again, the dataset is still not made publicly available yet. If you need this file, please contact Jeremy.
# Roadmap/Todos
- Implement a command line interface to specify the network architecture (e.g. Alex-Net-like CNN, fully connected ANN, etc.) and optimization techniques (e.g. Adam, SGD, and hyperparameters). Status: **SOMEWHAT COMPLETED**.
- Implement a logging system for effective debugging. Status: **SETUP COMPLETED**.
- Figure out the data distribution and what to do with them (w.r.t. to shared images, partially shared images, and unique images). Status: **COMPLETED, decided to do imputation.**
- Brief preliminary visualization of training data. Status: **COMPLETED**. This task was carried out to determine if those pictures that the subject has seen multiple time are spread out significantly. If they are not spread out too much (i.e. the activations are consistent), we can just take the mean, which can reduce the training samples by 3 times. The relevant graphs generated are in the `graphs` folder. The results showed that the data are quite spread out, and the spread of activation is similar in all subjects whether it is the shared image set or the entire image set. Furthermore, as per Zijin's advice, it is desirable to get all response as the training data to prevent underfitting.
- Investigate how useful the AlexNet feature space are as training data and find good hyperparameter for each of them. Status: **COMPLETED for NN and Linear Models**. Cross-validation that is going to be used to look for hyperparameters are going to be in the `experiments` folder. Each try at cross-validation will have its related data in the following path: `experiments/cross_validation_id/about.json`, where `id` is the timestamp where then cross-validation experiment dataset is created. The `about.json` for an incomplete cross-validation would have the `completed` attribute set to `False`. It will also have another attribute `partitions`, a list of JSON objects each holding the training data set (`train`) and the testing data set (`test`). The length of `partitions` should be the same as the number of folds we are doing in this k-fold CV. The state of the `about.json` of a completed CV experiment should have the `completed` attribute set to `True`, and the `partitions` attribute should remain unchanged. It will also have a `description` attribute to describe the CV, and an `errors` attribute to record the MSE of the images through the process. There should also be an `avg_progress.png` graph that plots the average training and testing MSE as detailed in the `errors` attribute of `about.json` against epoch.
- Imputation of data through kNN + SVD. Status: **INCOMPLETE, but LOW PRIORITY**.
- Set apart the validation data set. Status: **COMPLETED**. The input to the training dataset is generated in the `Custom_Dataset` class (a custom PyTorch Dataset).
- Investigate AlexNet feature space as a means of imputation and dimensionality reduction. Status: **ON HOLD**. AlexNet (the slightly modified one of the one implemented in Zijin's NeuroGen) takes in a 4D PyTorch Tensor of shape (n, 3, 227, 227), where n is the number of images. The output is 9 tensors, which is the output from each of the 5 convolutional layers, the input to the first fully connected layers, and the output of the 3 fully connected layers. The shapes are:
    * (n, 64, 27, 27)
    * (n, 192, 27, 27)
    * (n, 384, 13, 13)
    * (n, 256, 13, 13)
    * (n, 256, 13, 13)
    * (n, 9216)
    * (n, 4096, 1, 1)
    * (n, 4096, 1, 1)
    * (n, 1000, 1, 1)
    We still need to figure out what to use as the feature space (probably the flattened input).
- Add option to suppress sanity checks/assertions. Status: **PARTIALLY COMPLETE in main training function (`verbose` parameter)**.
- Find out all the NaN values and figure out what to do with them. Status: **COMPLETED**. Some subjects have certain ROIs missing, so their activation is NaN. The precise list is as follows:
    | ROI       | Subjects that do NOT have this ROI |
    |-----------|------------------------------------|
    | aTLfaces  | 4                                  |
    | FBA1      | 2, 7                               |
    | mTLbodies | 1, 2, 3, 7                         |
    | mTLfaces  | 1, 2, 3, 6                         |
    | mTLwords  | 3                                  |
    
    The data in the `validation` folders are the mean activation among those subjects that do have the certain ROI.  
- Pearson Correlation and violin plot. Status: **PARTIALLY COMPLETE**. Can now plot pearson correlation over epoch.
- Investigate why correlation and MSE is sometimes nan. Status: **INCOMPLETE**.  
- Optimize with Pearson Correlation instead of MSE. Status: **EXPLORED. FURTHER EXPLORATION ON HOLD.**. Added option to use Pearson correlation as the loss function. However, just using Pearson Correlation blows up the output (MSE would become huge). Some regularization is needed.  
- Use Neurogen's fwrf model to test for baseline correlation. Status: **COMPLETED**. Relevant code are in `baseline_exp.py` and `baseline`. The code in `baseline` are minimally edited code from Zijin's NeuroGen. The results obtained is shown below:
    | ROI           | Activation  |
    |---------------|-------------|
    | EBA           | 0.745624448 |
    | FBA1          | 0.560971152 |
    | FBA2          | 0.717303066 |
    | FFA1          | 0.762078177 |
    | FFA2          | 0.754860623 |
    | L_amygdala    | N/A         |
    | L_hippocampus | N/A         |
    | OFA           | 0.698782423 |
    | OPA           | 0.729941226 |
    | OWFA          | 0.620909915 |
    | PPA           | 0.820451649 |
    | RSC           | 0.773951489 |
    | R_amygdala    | N/A         |
    | R_hippocampus | N/A         |
    | V1d           | 0.861181372 |
    | V1v           | 0.879067739 |
    | V2d           | 0.772756183 |
    | V2v           | 0.850069941 |
    | V3d           | 0.733561857 |
    | V3v           | 0.785885334 |
    | VWFA1         | 0.539793891 |
    | VWFA2         | 0.573723877 |
    | aTLfaces      | 0.620114526 |
    | hV4           | 0.728416874 |
    | mTLbodies     | 0.368583526 |
    | mTLfaces      |  0.44080768 |
    | mTLwords      | 0.428752137 |
    | mfswords      | 0.517553413 |

    If we count the predicted activation for regions that fwRF can't handle as 0 (divide by 28, the number of regions), the mean correlation would be about 0.582. However, if we only look at the regions for which the fwRF model can predict (24 of them), the mean correlation would be about 0.679.  

- Transform to region-specific model instead. Status: **COMPLETED**. Added functionality to train region-specific model.
- Consider using feature maps other than AlexNet trained on ImageNet. Status: **INCOMPLETE, but LOW PRIORITY**.
- Explore using Linear models. Status: **COMPLETED**. Explored linear models on AlexNet 9216 feature maps. Results are poor compared to the neural networks. As the input are raised to higher power, the fit becomes worse in terms of both training and validation dataset.
- Explore fwRF models. Status: **IN PROGRESS**. Implemented fwRF model, but it is still quite buggy. Need Zijin's further assistance.
- Set aside testing set. Status: **COMPLETED**. Read from `all_images_related_data/partition.json` to get the image IDs of the images in the image sets.
- Implement model, optimizer, and loss wrappers API. Status: **COMPLETED**. The following is a brief usage example.
    ```
    import torch

    # import necessary modules written specifically for this project
    from model_wrappers.linear_model_wrapper import Average_Model_Regression, Linear_Model_Dataset, ridge_regression_loss
    from wrapper import Wrapper
    import main

    model = Average_Model_Regression(2)
    model_name = "lin_1"
    optim = torch.optim.Adam
    optim_params = [{'params': model.parameters(), 'lr': 2e-5}]
    optim_name = "adam_2e-5"
    dataset_class = Linear_Model_Dataset
    dataset_name = 'test'
    loss_func = lambda a, b: ridge_regression_loss(a, b, model, 0.1)

    # an instance of the Wrapper class
    model_wrapper = Wrapper(model=model, model_name=model_name, optim=optim, optim_params=optim_params, optim_name=optim_name, dataset_class=dataset_class, dataset_name=dataset_name, loss_func=loss_func, roi='EBA')

    # train using the Wrapper instance
    main.train_model(model_wrapper, epoch = 200)
    ```
- Grid Search for optimal x, y, sigma. Status: **INCOMPLETE**.
- Find the 1024 feature maps with highest variance for the fully connected layers. Status: **INCOMPLETE**.
- Wrapper `about.json` generation. Status: **INCOMPLETE**.