import numpy as np
import os
import random
import h5py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import scipy.io as scio


class MatHandler(object):    
    """
    Class for managing the dataset
    """

    def __init__(self, is_oneD_Fourier):    

        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.split_dataset(is_oneD_Fourier)

    def read_mat(self):
        """
        Read the .mat files in the oneD folder and return the data and labels
        It traverses all .mat files in the oneD folder, reads the data, processes it into segments of length 1024, and assigns labels based on the file names.
        """
        # DONE
        data = None        
        label = np.array([], dtype=int)
        count = 0
        # Traverse each mat file in the oneD folder
        for fn in os.listdir('oneD'):            
            if fn.endswith('.mat'):                         
                # Path 
                path = 'oneD/'+"".join(fn)                
                read_data = scio.loadmat(path)               
                # Get labels 
                now_data_label = fn.split('_')[0]          
                # print(now_data_label)
                # Get the list of dictionary keys in the mat file
                var_dict = list(read_data.keys())
                # Find the variable with 'DE' in the .mat file                
                for var in range(len(var_dict)):        
                    check_DE = var_dict[var].split("_")
                    for check in check_DE:
                        if check == 'DE':
                            # Record the position of DE
                            location = var
                            # Record the variable name with DE
                            var_DE = var_dict[location]
                            break

                # Read the data and transpose it
                now_data = read_data[var_DE].T                 
                # Remove the trailing part
                unwanted = now_data.shape[1] %1024   
                now_data = now_data[...,:-unwanted]
                # Split the data into 1024
                
                
                now_data = now_data.reshape(-1,1024) 
                now_data_len = now_data.shape[0]  
                     
                # Record labels
                for layer in range(int(now_data_len)):
                    label = np.append(label, int(now_data_label))
                # First record
                if count == 0:
                    data = now_data
                    count += 1
                    continue
                # Record more than twice
                data = np.vstack((data,now_data))
                count += 1
        # Return the dataset's data and labels
        print("Now data shape:", now_data.shape)
        data = data.reshape(-1, 1, 1024) 
        print("now data shape after reshape: ",now_data.shape) 
        return data, label

    def split_dataset(self, is_oneD_Fourier, scale = False ):
        """
        Load the dataset
        Split the dataset into training set, validation set, and test set
        """
        # DONE
        X, y = self.read_mat()
        #print(f"Total data shape: {X.shape}, Total labels shape: {y.shape}, Unique labels in the dataset: {np.unique(y)}, Labels distribution in X: {np.bincount(y)}")

        # Extract 30% of the data as the test set, and from that, extract 50% as the validation set
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)       # random state schanged from 30
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp) # random state schanged from 30
        print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        print(f"Train Labels: {np.unique(y_train)}, distrubution {np.bincount(y_train)}")

        X_train = np.squeeze(X_train)     # remove single-dimensional entries from the shape of an array (913, 1, 1024) -> (913, 1024)
        X_test = np.squeeze(X_test)  
        X_val = np.squeeze(X_val) 

        if scale == True:
            # Standardize features 
            scaler = StandardScaler()    
            """
            This is a class from scikit learn used for standardize features by removing the mean and scaling to unit variance. 
            This involves rescaling the features so that they have a mean of 0 and a standard deviation of 1 
            Most of ML algorithms perform better when the features are on the similar scale 
            """   
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_val = scaler.transform(X_val)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        y_val = np.squeeze(y_val)

        X_train = X_train.reshape(-1, 1, 1024) # (913, 1024, 1)
        X_test = X_test.reshape(-1, 1, 1024) # (196, 1024, 1)
        X_val = X_val.reshape(-1, 1, 1024) # (196, 1024, 1)

        # One-dimensional Fourier transform
        if is_oneD_Fourier == True:
            X_train = oneD_Fourier(X_train)
            X_test = oneD_Fourier(X_test)
            X_val = oneD_Fourier(X_val)

        # The output is written this way for convenience, so that it is easy to modify when using the training set, test set, and validation set in this experiment
        # Originally: X_train is the training set, X_test is the test set, X_val is included in X_train
        return X_train, y_train, X_val, y_val, X_test, y_test

def preprocessing(x):
    """
    TODO: No processing yet implemented
    """
    return x

def oneD_Fourier(data):
    """
    1D Fourier Transform
    """

    # The data has an extra dimension
    data = np.squeeze(data)

    for sample in range(data.shape[0]): # in case of training there is 913 samples
        data[sample] = abs(np.fft.fft(data[sample]))
    data = data.reshape(-1,1, 1024)
    
    return data

def get_Data_By_Label(mathandler = MatHandler(is_oneD_Fourier = False), pattern = 'train', label_list = [1,2,3,4,5,6,7,8,9]):
    """
    Get the dataset by label
    Label 0 is normal data, and other labels are fault data
    """
    if 'train' == pattern:
        data = mathandler.X_train
        label = mathandler.y_train
    elif 'test' == pattern:       
        data = mathandler.X_test
        label = mathandler.y_test
    elif 'val' == pattern:
        data = mathandler.X_val
        label = mathandler.y_val
    elif 'full' == pattern: 
        data = np.vstack((mathandler.X_train, mathandler.X_test, mathandler.X_val))
        label = np.hstack((mathandler.y_train, mathandler.y_test, mathandler.y_val))
    else:
        data = np.vstack((mathandler.X_train, mathandler.X_val))
        label = np.hstack((mathandler.y_train, mathandler.y_val))

    # Separate normal data
    idx_normal = np.where(label == 0)[0]
    data_normal = data[idx_normal]
    label_normal = label[idx_normal]
  
    # Separate data by label
    for i in label_list:
        idx = np.where(label == i)[0]
        data_temp = data[idx]
        label_temp = label[idx]
        data_normal = np.vstack((data_normal, data_temp))
        label_normal = np.hstack((label_normal, label_temp))

    # Set the random seed so that the dataset can be reproduced
    random.seed(1)

    # Shuffle the dataset
    index = [i for i in range(len(data_normal))]
    random.shuffle(index)
    data_normal = data_normal[index]
    label_normal = label_normal[index]
    return data_normal, label_normal

def load_Dataset_Original(
    batch_size = 1, 
    is_oneD_Fourier = False,
    pattern = 'full',
    label_list = [0,1,2,3,4,5,6,7,8,9]
    ):
    
    data, labels = get_Data_By_Label(
        mathandler = MatHandler(is_oneD_Fourier = is_oneD_Fourier), 
        pattern = pattern, 
        label_list = label_list
        )
    
    dataset = torch.tensor(data, dtype=torch.float64)
    print(dataset.shape)
    label = torch.tensor(labels, dtype = torch.float64) 
    print(label.shape)
    return dataset

def save_data(data = MatHandler(is_oneD_Fourier = False),format = 'npy'):
    """
    Save the processed data to a file
    """
    # Ensure "data" directory exists
    os.makedirs("data", exist_ok=True)

    if format == 'npy':
        np.savez_compressed("data/dataset.npz", 
                            X_train=data.X_train, y_train=data.y_train,
                            X_val=data.X_val, y_val=data.y_val,
                            X_test=data.X_test, y_test=data.y_test)
        #data = np.load("dataset.npz")
        # X_train = data['X_train']
    elif format == 'tensor':
        print("Not impplemented")
    elif format == 'h5py':
        with h5py.File("data/dataset.h5", "w") as f:
            f.create_dataset("X_train", data=data.X_train)
            f.create_dataset("y_train", data=data.y_train)
            f.create_dataset("X_val", data=data.X_val)
            f.create_dataset("y_val", data=data.y_val)
            f.create_dataset("X_test", data=data.X_test)
            f.create_dataset("y_test", data=data.y_test)
    else:
        raise ValueError("Unsupported format. Use 'npy' or 'mat'.")

if __name__ == "__main__":
    """
    Test the effect of dataset generation
    """
    data, label = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),
                                    pattern='train',
                                    label_list=[]
                                    ) 

    print(type(data), data.dtype, data.shape)

    #data   = np.ascontiguousarray(data)
    print(type(data), data.dtype, data.shape)

    #labels = np.ascontiguousarray(label)
    X = torch.tensor(data, dtype=torch.float64)  # copies data
    y = torch.tensor(label, dtype=torch.long)    # copies data

    print("Shape of the X is :", X.shape)                         # no parentheses
    print(y.shape)

    train_dataset = TensorDataset(X)
    train_loader = DataLoader(train_dataset)