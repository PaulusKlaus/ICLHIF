import tensorflow as tf
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
import scipy.io as scio


# at top
try:
    import tensorflow as tf
    _HAVE_TF = True
except Exception:
    _HAVE_TF = False

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
        data = data.reshape(-1, 1024, 1)  
        return data, label

    def split_dataset(self, is_oneD_Fourier):
        """
        Load the dataset
        Split the dataset into training set, validation set, and test set
        """
        # DONE
        X, y = self.read_mat()

        # Extract 30% of the data as the test set, and from that, extract 50% as the validation set
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)       # random state schanged from 30
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

        X_train = np.squeeze(X_train)     # remove single-dimensional entries from the shape of an array (913, 1024, 1) -> (913, 1024)
        X_test = np.squeeze(X_test)  # (196, 1024, 1) -> (196, 1024)
        X_val = np.squeeze(X_val) # (196, 1024, 1) -> (196, 1024)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        y_val = np.squeeze(y_val)


        X_train = X_train.reshape(-1, 1024, 1) # (913, 1024, 1)
        X_test = X_test.reshape(-1, 1024, 1) # (196, 1024, 1)
        X_val = X_val.reshape(-1, 1024, 1) # (196, 1024, 1)

        # One-dimensional Fourier transform
        if is_oneD_Fourier == True:
            X_train = oneD_Fourier(X_train)
            X_test = oneD_Fourier(X_test)
            X_val = oneD_Fourier(X_val)


        # The output is written this way for convenience, so that it is easy to modify when using the training set, test set, and validation set in this experiment
        # Originally: X_train is the training set, X_test is the test set, X_val is included in X_train
        return X_train, y_train, X_val, y_val, X_test, y_test#, X_train_val, y_train_val       # cheng：X_train_val为训练集 + 测试集、 X_train_val and y_train_val add by cheng

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
    data = data.reshape(-1,1024,1)
    
    return data

def get_Data_By_Label(mathandler = MatHandler(is_oneD_Fourier = False), pattern = 'train', label_list = [1, 2, 3]):
    """
    Get the dataset by label
    Label 0 is normal data, and other labels are fault data
    Label 1: 
    Label 2:
    Label 3:
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
    label_list = [1,4,7], 
    batch_size = 1, 
    is_oneD_Fourier = False,
    pattern = 'train'
    ):


    data, labels = get_Data_By_Label(
        mathandler = MatHandler(is_oneD_Fourier = is_oneD_Fourier), 
        pattern = pattern, 
        label_list = label_list
        )
    
    # Get data from the oneD folder
    if not _HAVE_TF:
        # Fallback: just return numpy arrays when TF is unavailable
        print("TensorFlow is not available. Returning numpy arrays.")
        return data

    AUTO = tf.data.experimental.AUTOTUNE # tune the performance of input pipeline
    dataset = tf.data.Dataset.from_tensor_slices(data) # create a dataset from tensor slices
    dataset = dataset.shuffle(1024).map(preprocessing, num_parallel_calls=AUTO).batch(batch_size).prefetch(AUTO) # prefetch data for better performance
    # from_tensor_slices -> shuffle(1024) -> map(add1) -> batch(batch_size) (each batch is (batch_size,1024,1)) -> prefetch(AUTOTUNE)
    return dataset

if __name__ == "__main__":
    """
    Test the effect of dataset generation
    """
    data, label = get_Data_By_Label(label_list=[0], pattern='full')    
    
    print(data) 
    print(label)
    print(data.shape)
    print(label.shape)
    print('suc')