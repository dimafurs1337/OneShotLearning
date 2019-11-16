import numpy as np
from mnist import MNIST
from keras.models import load_model
from sklearn.metrics import accuracy_score

def load_data():
    mnist = MNIST('C:/Users/My/Desktop/OneShot/mnist/')
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
	

	
	
	
	
	