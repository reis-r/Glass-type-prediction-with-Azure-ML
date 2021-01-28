import argparse
import os
import joblib
import numpy as np
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run
from sklearn.model_selection import train_test_split

# Select the data URL
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'

# create an output folder for the model file
if "outputs" not in os.listdir():
    os.mkdir("./outputs")

ds = TabularDatasetFactory.from_delimited_files(path=data_url, header=False)

def clean_data(data):
    # Transform the TabularDataset into a pandas dataframe, removing NA's
    x_df = data.to_pandas_dataframe().dropna()
    
    # Rename the columns to it's right labels
    x_df.columns = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    
    # Make the target variable the right type
    x_df.Type = x_df.Type.astype('category')
    
    y_df = x_df.pop("Type")
    
    return x_df, y_df

# Apply data cleaning
x, y = clean_data(ds)

# Split train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

run = Run.get_context()

def main(): 
    # Set script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='Specifies the kernel type to be used in the algorithm')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter')
    
    # Parse the command line argumments
    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Regularization parameter', np.float(args.C))
    
    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel=args.kernel, C=args.C).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)
    
    # model evaluation for X_test
    accuracy = svm_model_linear.score(x_test, y_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')


# Run the main script method if it's the case
if __name__ == '__main__':
    main()