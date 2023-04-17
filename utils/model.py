import numpy as np
import os 
import joblib


class Perceptron:
    def __init__(self ,eta: float=None ,epochs : int=None):
        self.weights = np.random.randn(3) * 1e-4 # small random weights
        training = (eta is not None) and (epochs is not None)
        if  training:
            print(f"intial weights before training : {self.weights}")
            
        self.eta = eta
        self.epochs = epochs
            
    # function starting with _ cant be access outside of class
    def _z_output(self , inputs, weights):
        return np.dot(inputs ,weights)
        
    
    def activation_function(self ,z):
        # activation finction will give 1 if z > 0 else 0
        return np.where(z >0 , 1, 0)
    
    def fit(self ,X ,y):
        self.X = X
        self.y = y
        
        # we are concatenating X with bias
        X_with_bias = np.c_[self.X ,-np.ones((len(self.X) ,1))]
        print(f"X with bias:\n {X_with_bias}")
        
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch >>{epoch}")
            print("--"*10)
            
            z = self._z_output(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            print(f"Predicted value after forward pass :\n{y_hat}")
            
            self.error = self.y - y_hat
            print(f"Error : \n{self.error}")
            
            # updating the weights
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T ,self.error)
            print(f"updated weights after epoch :{epoch +1 }/{self.epochs} : \n {self.weights}")
            print("##"*10)
    
    def predict(self ,X ):
        X_with_bias = np.c_[X ,-np.ones((len(X) ,1))]
        z = self._z_output(X_with_bias ,self.weights)
        return self.activation_function(z)
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\ntotal loss : {total_loss}\n")
        return total_loss
        
    def _create_dir_return_path(self ,path ,filename):
        os.makedirs(path ,exist_ok = True)
        return os.path.join(path ,filename)
    
    def save(self ,filename ,path = None):
        if path is not None:
            model_file_Path = self._create_dir_return_path(path ,filename)
            joblib.dump(self ,model_file_Path)
        else:
            model_file_Path = self._create_dir_return_path("model" ,filename)
            joblib.dump(self ,model_file_Path)  
            
    def load(self ,filepath):
        return joblib.load(filepath)