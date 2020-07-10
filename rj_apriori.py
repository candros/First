import pandas as pd
from apyori import apriori
import pickle

class mineral_apriori():
    def __init__(self,input_file,support,output):
        
        # Load a pickled list of minerals grouped by location found.
        with open(input_file,'rb') as f:
            self.minerals = pickle.load(f)
            
            # Calculating the association rules for the inputted minerals.
            # The support value for the most uncommon mineral found in the 
            # Mindat database is: 0.00022 
            association = apriori(self.minerals,min_support = support,min_length = 2)
            self.results = list(association)
            self.output = output + '.csv'
            
    def write_rules(self):  
        # Extract the values from the apriori output and package them in a
        # DataFrame and export it to a .csv file.
        df = []
        for i in range(len(self.results)):
            group = [list(self.results[i].items), self.results[i].support,
                     self.results[0].ordered_statistics[0][2], self.results[0].ordered_statistics[0][3]]
            df.append(group)
        df = pd.DataFrame(df,columns = ['minerals','support','confidence','lift'])
        df.to_csv(self.output)