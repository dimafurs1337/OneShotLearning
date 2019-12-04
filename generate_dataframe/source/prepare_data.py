import os
import sys
import numpy as np
from mnist import MNIST

data_path = "generate_dataframe/data/"

def load_data():
    #os.chdir("..")
    #path = os.getcwd() + "/data/"
    mnist = MNIST(data_path)
    x_train, y_train = mnist.load_training() 
    x_test, y_test = mnist.load_testing()
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)
    return x_train/255, y_train, x_test/255, y_test

def create_data(n_samples, x, y):
    data = np.zeros([n_samples, 2, 1, 28, 28])  
    labels = np.zeros([n_samples, 1])  
    for k in range(n_samples//2):
        i, j = np.random.randint(1, y.shape[0]), np.random.randint(1, y.shape[0])     
        while y[i] != y[j]:
            i, j = np.random.randint(1, y.shape[0]), np.random.randint(1, y.shape[0]) 
        data[k, 0, 0, :, :] = x[i].reshape(28,28)
        data[k, 1, 0, :, :] = x[j].reshape(28,28)        
        labels[k] = 1        
    for k in range(n_samples//2, n_samples):
        i, j = np.random.randint(1, y.shape[0]), np.random.randint(1, y.shape[0])         
        while y[i] == y[j]:
            i, j = np.random.randint(1, y.shape[0]), np.random.randint(1, y.shape[0])         
        data[k, 0, 0, :, :] = x[i].reshape(28,28)
        data[k, 1, 0, :, :] = x[j].reshape(28,28)        
        labels[k] = 0
            
    return data, labels
	
def get_data(train_size=10000, test_size=5000):
	x_train, y_train_labels, x_test, y_test_labels = load_data()
	x_train, y_train = create_data(train_size, x_train, y_train_labels)
	x_test, y_test = create_data(test_size, x_test, y_test_labels)
	return x_train, y_train, x_test, y_test
	
def load_existing_data(path):
	data = np.load(path)
	return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']
	
#if __name__ == "__main__":
    #train_size, test_size = int(sys.argv[1]), int(sys.argv[2])
    #data = get_data(train_size, test_size)
    #os.chdir("generated_data")
    #np.savez("generater_data.npz", X_train=data[0], Y_train=data[1], X_test=data[2], Y_test=data[3])	


os.chdir("C:\\Users\\My\\Desktop\\OneShot\\project")
data = get_data(10000, 5000)
np.savez("C:\\Users\\My\\Desktop\\OneShot\\project\\generate_dataframe\\generated\\generated_data.npz", X_train=data[0], Y_train=data[1], X_test=data[2], Y_test=data[3])	