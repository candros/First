from scipy.io import loadmat
import pandas as pd
import numpy as np
import pickle

#This class function takes a *.mat file, containing a struct called "stDataAll"
#and a pickled model file called "tourmaline_model". The function
#predict_location provides a list labeling each input as either Brazil or
#Africa. The predict function provides a dataframe with the sample names, their
#predicted location and the probability that the sample is either African or 
#Brazilian.
class tourmaline_model():
    
    def __init__(self,data_file,model_file):
        
        #Loading the mat input file
        mat = loadmat(data_file)
        self.data = mat['stDataAll']
        
        #loading the logistic regression model
        with open('tourmaline_model','rb') as model_file:
            self.model = pickle.load(model_file)
            
        #Extracting the names of the samples
        samples_raw = self.data[0][0][0][:]
        samples = [samples_raw[i][0][0] for i in range(len(samples_raw))]
        self.results = pd.DataFrame(samples,columns = ['Sample Names'])
        
        #Getting the 15 features we selected using recursive feature analysis
        feature_names = ['279.632', '280.282', '288.2', '309.381', '313.067', '313.086',
       '313.105', '313.124', '313.143', '324.778', '324.798', '396.166',
       '396.191', '396.242', '481.151']
        self.features = feature_names
        AllFeatures = pd.DataFrame(self.data[0][0][1][:])
        wavelength_names = self.data[0][0][4][0]
        wavelength_index = []
        for x in feature_names:
            ind = [i for i in range(len(wavelength_names)) if np.float(x) == wavelength_names[i]]
            wavelength_index.append(ind[0])
        self.inputs = AllFeatures.iloc[:,wavelength_index]
            
    def predict_location(self):
        if (self.inputs is not None):
            pred_outputs = self.model.predict(self.inputs)
            pred_location = ['Brazil' if x == 1 else 'Africa' for x in pred_outputs]
            return pred_location

    # predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def predict(self):
        if (self.inputs is not None):
            location = self.model.predict(self.inputs)
            self.results['Predicted Location'] = ['Brazil' if x == 1 else 'Africa' for x in location]
            self.results['Probability Brazil'] = self.model.predict_proba(self.inputs)[:,1]
            self.results['Probability Africa'] = self.model.predict_proba(self.inputs)[:,0]
            return round(self.results,4)
