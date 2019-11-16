import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam, Nadam
from keras.layers import Dense, Dropout, MaxPooling2D, Input, Flatten, Lambda, Conv2D
import keras.backend as K
from sklearn.metrics import accuracy_score


class SiameseNetwork:
    margin = 1

    def __init__(self, input_shape):       
        img1 = Input(shape=input_shape)
        img2 = Input(shape=input_shape)    
        
        model = Sequential()
	
        model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape, 
                     data_format='channels_first', kernel_initializer='random_uniform', 
                     bias_initializer='zeros'))    
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='random_uniform', 
                     bias_initializer='zeros'))
    
        model.add(MaxPooling2D())
        model.add(Dropout(.25))
        model.add(Conv2D(128, (2,2), activation='relu', kernel_initializer='random_uniform', 
                     bias_initializer='zeros'))
    
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='sigmoid'))    
    
        features1 = model(img1)
        features2 = model(img2)
    
        L1_layer = Lambda(lambda vectors:K.abs(vectors[0] - vectors[1])) 
        L1_distance = L1_layer([features1, features2])
        prediction = Dense(1,activation='sigmoid')(L1_distance)
        self.model = Model(inputs=[img1,img2],outputs=prediction)

    @staticmethod
    def contrastive_loss(y_true, y_pred, margin=margin):
	    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    def make_prediction(self, x_test):
        return self.model.predict([x_test[:,0], x_test[:,1]])

    def fit(self, x_train, y_train, batch_size, epochs):
        ndm = Nadam()
        self.model.compile(loss=self.contrastive_loss, optimizer=ndm)
        img_1 = x_train[:,0]
        img_2 = x_train[:,1]
        self.model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=batch_size, verbose=1, epochs=epochs)

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, name):
        self.model.save(name)


class Accuracy:
    @staticmethod
    def compute_accuracy(y_test, prediction):
        def filter_values(x): 
            if x < 0.5: return 1 
            else: return 0
        filter_values = np.vectorize(filter_values)
        prediction = filter_values(prediction)
        return accuracy_score(y_test, prediction)

    @staticmethod
    def compute_probabilities(prediction):
        return 1-prediction






