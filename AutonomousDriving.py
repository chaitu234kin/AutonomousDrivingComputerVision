import numpy as np
import cv2
from tqdm import tqdm
import time
from scipy import optimize
import math  

def normalize_image(image) :
    resized_image = cv2.resize(image,(60, 64))
    cropped_image = resized_image[50:,:,0]
    grey_scale_threshold_image_boolean = (cropped_image > 220)
    return grey_scale_threshold_image_boolean
     
def returnFrameandSteeringangles(csv):
    data = np.genfromtxt(csv, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    return [frame_nums, steering_angles]

def getCentres_Indexes_ofangles_atbins(steering_angles):
    bins = 64
    centers = np.linspace(-165, 165, bins)
    index_ofangles_AT_BINS =np.digitize(steering_angles,centers,right=True)
    return [centers, index_ofangles_AT_BINS]

def nearest_bin_to_steeringAngle(centers, indexofangles_atbins, steering_angle, index_of_angle):
    index_of_current_angle = indexofangles_atbins[index_of_angle]
    left_most_bin_index = index_of_current_angle - 1
    rightmost_nearest_bin = centers[index_of_current_angle]
    leftmost_nearest_bin = centers[left_most_bin_index]
    current_steering_angle = steering_angle[index_of_angle]
    
    if (abs(leftmost_nearest_bin - current_steering_angle) > abs(rightmost_nearest_bin - current_steering_angle)):
        return index_of_current_angle
    else:
        return left_most_bin_index
    
def trim_left_pushWeightedArray(zero_weight_array, nearest_bin_index, gaussianDistributed_weights, total_elements_toleft, total_elements_toright_including_1):
        elements_to_trim = total_elements_toleft-nearest_bin_index 
        zero_weight_array[0:nearest_bin_index+total_elements_toright_including_1] = gaussianDistributed_weights[elements_to_trim:]
        return zero_weight_array
    
def trim_right_pushWeightedArray(zero_weight_array, nearest_bin_index, gaussianDistributed_weights, total_elements_toleft, total_elements_toright_including_1):
        elements_to_trim = 64-nearest_bin_index 
        zero_weight_array[nearest_bin_index-total_elements_toright_including_1:] = gaussianDistributed_weights[0:elements_to_trim+total_elements_toright_including_1]
        return zero_weight_array

def pushTheWeightedArray(zero_weight_array, nearest_bin_index, gaussianDistributed_weights, total_elements_toleft, total_elements_toright_including_1):
        zero_weight_array[nearest_bin_index-total_elements_toleft:nearest_bin_index+total_elements_toright_including_1] = gaussianDistributed_weights
        return zero_weight_array
    
def get_directionToTrim(nearest_bin_index, total_elements_toleft, total_elements_toright_including_1):
    if (nearest_bin_index < total_elements_toleft):
        return "Left"
    elif (nearest_bin_index+total_elements_toright_including_1 > 64):
        return "Right"
    else:
        return "None"
    
def fillzerosArray_withGaussianDistributedWeights(nearest_bin_index, gaussianDistributed_weights):
    zero_weight_array = np.zeros(64)     
    total_elements_toleft = round(len(gaussianDistributed_weights)/2)-1
    total_elements_toright_including_1 = len(gaussianDistributed_weights) - total_elements_toleft
    
    direction_toTrim = get_directionToTrim(nearest_bin_index, total_elements_toleft, total_elements_toright_including_1)
    
    if (direction_toTrim == 'Left'):
        zero_weight_array = trim_left_pushWeightedArray(zero_weight_array, nearest_bin_index, gaussianDistributed_weights, total_elements_toleft, total_elements_toright_including_1)
    elif (direction_toTrim == 'Right'):
        zero_weight_array = trim_right_pushWeightedArray(zero_weight_array, nearest_bin_index, gaussianDistributed_weights, total_elements_toleft, total_elements_toright_including_1)
    else:
        zero_weight_array = pushTheWeightedArray(zero_weight_array, nearest_bin_index, gaussianDistributed_weights, total_elements_toleft, total_elements_toright_including_1)
        
    return zero_weight_array
    
    
def train(path_to_images, csv_file):
    data_frame_angles = returnFrameandSteeringangles(csv_file)
    frame_nums = data_frame_angles[0]
    steering_angles = data_frame_angles[1]
    
    NN = NeuralNetwork()
    List_of_images = []
    
    # normalize the output by assigning each output steering angle to a certain index of bins which are 64 divided from -165 to 165.
    centres_index_bins = getCentres_Indexes_ofangles_atbins(steering_angles)
    centers = centres_index_bins[0]
    indexofangles_atbins = centres_index_bins[1]
    
    list_of_zerobins_images = []
    gaussianDistributed_weights = [0.2, 0.24, 0.46, 0.68, 0.90, 1, 0.90, 0.68, 0.46, 0.24, 0.2]
    for index_of_angle in range(0,len(steering_angles)) :
        image = cv2.imread(path_to_images + '/' + str(int(index_of_angle)).zfill(4) + '.jpg') # read image one by one.
        normalized_image = normalize_image(image)
        normalized_flattened_image = normalized_image.ravel()
        
        List_of_images.append(normalized_flattened_image)
        
        nearest_bin_index = nearest_bin_to_steeringAngle(centers, indexofangles_atbins, steering_angles, index_of_angle)

        zeros = fillzerosArray_withGaussianDistributedWeights(nearest_bin_index, gaussianDistributed_weights)
        list_of_zerobins_images.append(zeros)
        
    X = np.array(List_of_images)
    Y = np.array(list_of_zerobins_images)
    NN.gradientDescent(X, Y, 0.01, 4000)
    return NN

def predict(NN, image_file):
    bins=64
    X=[]
    Bins_of_angles = np.linspace(-165, 165, bins)
    
    image_toPredict = cv2.imread(image_file)
    normalized_image = normalize_image(image_toPredict)
    normalized_flattened_image = normalized_image.ravel()
    X.append(normalized_flattened_image)
    X = np.array(X)
    
    yhat = NN.forward(X)
    return Bins_of_angles[np.argmax(yhat[0, :])]


class NeuralNetwork(object):
    def init_weights(self):
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)*np.sqrt(2/(self.inputLayerSize + self.hiddenLayerSize))
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)*np.sqrt(2/(self.hiddenLayerSize + self.outputLayerSize))
    
    def __init__(self):        
        #Define Hyperparameters
        # Number of input layers
        self.inputLayerSize = 840
        # Number of output nodes
        self.outputLayerSize = 64
        # Number of Hidden nodes
        self.hiddenLayerSize = 60
        self.init_weights()
              
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    def gradientDescent(self,X ,y, learning_rate, iterations):
        for each_iter in range(0,iterations):
            list_of_gradients = self.computeGradients(X,y)
            djW1= list_of_gradients[0]
            djW2 = list_of_gradients[1]
            self.W1 = self.W1 - learning_rate * djW1
            self.W2 = self.W2 - learning_rate * djW2

    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return [dJdW1, dJdW2]
