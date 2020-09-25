import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import sys

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
    	self.dmin = pd.Series()
    	self.dmax = pd.Series()

    def __call__(self, features, is_train=False):
    	if is_train:
    		self.dmin = features.min()
    		self.dmax = features.max()
    	features -= self.dmin
    	features /= (self.dmax - self.dmin)


def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''
    cols = np.arange(0,60)
    removed = [59]
    df = pd.read_csv(csv_path, skipinitialspace=True, usecols=[i for i in cols if i not in removed])
    scaler(df, is_train)

    return df.to_numpy()
    # raise NotImplementedError

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    return pd.read_csv(
        csv_path, 
        skipinitialspace=True, 
        usecols=["shares"]
    ).to_numpy()
    # raise NotImplementedError
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    # w = [(X'X + mCI)^-1] * X'Y

    weights = np.dot(feature_matrix.transpose(), feature_matrix)
    i = np.identity(weights.shape[1]) 
    weights = np.add(weights, feature_matrix.shape[0] * C * i)
    weights = np.linalg.inv(weights)
    x = np.dot(feature_matrix.transpose(), targets)
    weights = np.dot(weights, x)
    
    return weights
    # raise NotImplementedError 

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array of shape m x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''

    # f(x) = Xw

    return np.dot(feature_matrix, weights)
    # raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''

    # f(x) = Xw
    fx = np.dot(feature_matrix, weights)

    # loss = sumition (f(x) - y)^2 / m
    v = np.subtract(fx, targets)
    v = np.multiply(v, v)
    loss = np.sum(v)/v.shape[0]

    return loss
    # raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    #l2 norm of w = sqrt(w1^2 + w2^2 + w3^2 + ....)
    return np.linalg.norm(weights, 2)

    # raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    loss = mse_loss(feature_matrix, weights, targets) + C * l2_regularizer(weights)
    return loss
    # raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    # grad = -(2/m)*X'.(Y - Xw) + 2Cw 

    grad = -(2/feature_matrix.shape[0]) * feature_matrix.transpose()
    x = np.subtract(targets, np.dot(feature_matrix, weights))
    grad = np.dot(grad, x)
    grad = np.add(grad, 2*C*weights)
    return grad

    # raise NotImplementedError

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    x = np.random.randint(0, targets.shape[0] - batch_size)
    return (feature_matrix[x : x + batch_size], targets[x : x + batch_size])

    # raise NotImplementedError
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.random.rand(n, 1)
    # raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    
    # w* = w - (lr*g)
    return np.subtract(weights, lr * gradients)

    # raise NotImplementedError

def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
    # allowed to modify argument list as per your need
    # return True or False
    raise NotImplementedError
    

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows -- 
    '''
    weights = initialize_weights(train_feature_matrix.shape[1])
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

        '''
        implement early stopping etc. to improve performance.
        '''
        #if dev_loss < 0.10:
        #	print("***early_stopping***")
        #	break

    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    #predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

def test_set(weights,scaler):
    # function for predicting values for test set based on calculated weights
    # saves output in csv file
    df = pd.read_csv('data/test.csv', skipinitialspace=True)
    scaler(df, False)
    test_features = df.to_numpy()
    predictions = get_predictions(test_features, weights)
    index = np.arange(0,11894).reshape((11894,1))
    predictions = np.column_stack((index, predictions))
    np.savetxt('output.csv', predictions, "%d,%d", header="instance_id,shares",comments="")
    return

if __name__ == '__main__':
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e-9)
    print('evaluating analytical_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, a_solution)
    train_loss = do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    #test_set(a_solution,scaler)

    #sys.exit()

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(
    					train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.12,
                        C=1e-11,
                        batch_size=32,
                        max_steps=2000000,
                        eval_steps=10000)

    print('evaluating iterative_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss = do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    #test_set(gradient_descent_soln, scaler)
