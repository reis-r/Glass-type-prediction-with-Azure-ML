# Glass type prediction web service using Azure ML

This project aims to create a web service for a model trained using the Azure Machine Learning Python SDK. There will be two models trained, one using scikit-learn's SVC module with hyperparameter tunning and another using Azure AutoML. The model with best accuracy will be then deployed and consumed.

## Project Set Up and Installation

The project will use an Azure ML Workspace, with the Jupyter notebook environment inside Azure itself. This avoids any extra setup and module instalation.
There will be 3 main files on the project:
- A Jupyter Notebook with all the AutoML process, including deployment (`automl.ipynb`)
- A Jupyter Notebook with the Hyperdrive process (`hyperparameter_tuning.ipynb`)
- A python script for data treatment, as well as the scikit-learn model training (`train.py`)

Care was taken for the data to be treated in the same way in both model training. The other files in the repository were created during the process of running the notebooks.

## Dataset

### Overview
The data used for the training is the [UCI Glass Identification](https://archive.ics.uci.edu/ml/datasets/Glass+Identification) dataset used to predict glass type based on various factors. The ID column was removed, since it does not present relevant information and can lead to wrong results.

### Task
The task will be to predict the type of glass based on the data describing various characteristics of glass.

The data will be accessed directly using the web link found in the UCI database, then treated using the train.py.

## Automated ML
The AutoML training is executed with a timeout of one hour and four concurrent iterations. This allows for resource saving and fast development. It used 3 cross-validations for a good assessment of the results. Deep learning was disabled since the dataset isn't too complex and might not bennefit from it.

### Results
The accuracy obtained was of 79%, the best model was a VotingEnsemble. The full configuration of the model is as follows:
```
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                                 feature_sweeping_config=None,
                                 feature_sweeping_timeout=None,
                                 featurization_config=None, force_text_dnn=None,
                                 is_cross_validation=None,
                                 is_onnx_compatible=None, logger=None,
                                 observer=None, task=None, working_dir=None)),
                ('prefittedsoftvotingclassifier',...
                                                 min_samples_leaf=0.01,
                                                 min_samples_split=0.10368421052631578,
                                                 min_weight_fraction_leaf=0.0,
                                                 n_estimators=25,
                                                 n_jobs=1,
                                                 oob_score=False,
                                                 random_state=None,
                                                 verbose=0,
                                                 warm_start=False))],
                                                                     verbose=False))],
                                               flatten_transform=None,
                                               weights=[0.16666666666666666,
                                                        0.16666666666666666,
                                                        0.16666666666666666,
                                                        0.16666666666666666,
                                                        0.16666666666666666,
                                                        0.16666666666666666]))],
         verbose=False)
```

The results could probably improve having AutoML run for more time, but for our purposes this is good enough.

![RunDetails widget for the AutoML experiment](https://raw.githubusercontent.com/reis-r/nd00333-capstone/master/screenshots/RunDetails_automl.PNG)
![RunDetails widget for the AutoML experiment](https://raw.githubusercontent.com/reis-r/nd00333-capstone/master/screenshots/automl_results.PNG)

## Hyperparameter Tuning

For the hyperparameter tuning we choose scikit-learn's SVC module, the tuned hyperparameters were the kernel type (categorical) and the regularization parameter (a number that goes from 0 to 1).

### Results
The accuracy obtained for the model was 65.11%. The model probably could improved using Grid sampling instead of Random sampling, but that would take longer (and more computing power) to train.

![RunDetails widget for the Hyperdrive experiment](https://raw.githubusercontent.com/reis-r/nd00333-capstone/master/screenshots/RunDetails_hyperdrive.PNG)
![Results for the Hyperdrive experiment](https://raw.githubusercontent.com/reis-r/nd00333-capstone/master/screenshots/hyperdrive-results.PNG)
## Model Deployment
The deployed model was the AutoML model, which obtained better accuracy. The model was deployed to an Azure Container Instance (ACI). The ACI provides a fast deployment ideal for development situations. The deployment was made using authentication and with Application Insights enabled.

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
