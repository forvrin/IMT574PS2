# IMT 574
# Toby Bianchi
# Problem Set 2: Calculating Eigenfactor Scores

# Imports
import numpy as np 
import pandas as pd
import datetime
import csv


class EigenFactorFinder:
    const_alpha = 0.85
    const_epsilon = 0.00001
    boolTesting = False
    filepathstring = ""

    #Constructor
    def __init__(self):
        print("Initializing...")
        self.filepathstring = 'links.txt'
        time_start = datetime.datetime.now()
        print("Begin Processing: {}".format(time_start))
        M = self.Matrix()
        # set the diagonal to 0
        M = self.diagnolize_Matrix(M)
        H = self.normalize_Matrix(M)
        danglingNodeVector = self.identify_Dangles(H)
        articleVector = self.calculateArticleVector()
        startVector = self.initStartVector(self.boolTesting)
        influenceVector = self.genInfVector(H, danglingNodeVector, articleVector, startVector)
        ef = self.findEigenFactor(H, influenceVector)
        time_end = datetime.datetime.now()
        print("Time finished: {}".format(time_end))
        print("Time Taken: {}".format((time_end - time_start)))
        ef = pd.DataFrame(ef)
        ef = ef.sort_values(0, ascending=False)
        #Top 20 journals
        print(ef[:20])



    @classmethod
    def test_Matrix(cls):
        cls.boolTesting = True
        print("We are testing the algorithm. boolTesting is now set to TRUE")
        return cls()


    #Define Methods
    def Matrix(self):
        print("Making a Matrix!")
        if self.boolTesting == True:
            print("Entering test_Matrix()")
            #define the test matrix
            d = [[1, 0, 2, 0, 4, 3],
                 [3, 0, 1, 1, 0, 0],
                 [2, 0, 4, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1],
                 [8, 0, 3, 0, 5, 2],
                 [0, 0, 0, 0, 0, 0]]
            tMatrix = np.array(d, dtype = float)
            print(tMatrix)
            print("finished creating test Matrix.")
            return tMatrix
        else:
            print("Using Live Data!")
        
        raw_list = self.Load_File(self.filepathstring)
        print("File loaded!")

        print("Converting to integers")
        for line in raw_list:
            for item in line:
                item = int(item)
        print("Conversion Complete")
        mtx = np.zeros((10747, 10747))
        
        for source, cite, value in raw_list:
            mtx[int(source)-1][int(cite)-1] = int(value)
        print(type(mtx))
        print("Matrix Managed!")            
        return mtx
    def diagnolize_Matrix(self, df):
        #Make sure we remove entries where a journal is citing itself
        print("Diagnolizing")

        for x in range(0, df.shape[0]):
            df[x][x] = 0
        print("Done with Diagnolizing. Time to Normalize!")
        return df
    def normalize_Matrix(self, df):
        print("Normalizing.  No one let on that we're weird.")
        try:
            colsums = df.sum(axis = 0, skipna = True)
        except:
            colsums = df.sum(axis=0)
        #divide each value in a column by the total of that column
        x = 0
        while x < len(colsums):
            #y = 0
            #while y < df.shape[1]:
            #    numerator = df[y,x]
            df[:, x] = (df[:, x] / colsums[x])
            #    y+=1
            x+=1
        
        print(df)       
            
        #this will possibly result in NaNs due to dividing by 0
        #Thanks for not making me troubleshoot that, pandas!
        #print(any(df.isna()))
        df = np.nan_to_num(df)
        print("Done Normalizing.  I think we fooled them.")
        return df
    def identify_Dangles(self, df):
        print("Dingle-DangleTime!")
        dglVector = []
        x = 0
        for column in df.T:
            #print("{} sums up to {}".format(col, colsum))
            colsum = column.sum()
            if colsum > 0:
                dglVector.append(0)
            else:
                dglVector.append(1)
            x+=1
        dglVector = np.array(dglVector, dtype=float)
        print(dglVector)
        print("Dangles Dingled!")
        return dglVector
    def calculateArticleVector(self):
        print("We need an Article Vector!  Fire up the Calculamatron 5000!")
        if self.boolTesting == True:
            a_tot = 14
            a = {'A': 3, 'B': 2, 'C': 5, 'D': 1, 'E': 2, 'F': 1}
            #Normalize 
            for key in a.keys():
                a[key] = (int(a[key]) / a_tot)
        else:
            a_tot = 10747
            a = {}
            for i in range(0, a_tot):
                a[i] = 1 / a_tot
        print("OH NO! The Calculamatron 5000 has melted! But it looks like it finished first!")
        a = pd.DataFrame(a, index=[0])
        return a
    def initStartVector(self, boolTesting):
        print("Creating Start Vector")
        a = {}
        if boolTesting == True:
            print("Testing is True, n = 6")
            n = 6
            a = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1}
        else:
            print("Working with real data!  n = 10747")
            n = 10747
            for i in range(0, n):
                a[i] = 1
        for i in a.keys():
            a[i] = a[i] / n
        

        print("Start Vector Calculated!")
        a = pd.DataFrame(a, index=[0])
        return a
    def genInfVector(self, mtx, danglingNodeVector, articleVector, startVector):
    
        print("Influence Vectorizimication Engaged.  I hope we live through this.")           
        articleVector = articleVector.transpose()
        startVector = startVector.transpose()
        #mtx = mtx.fillna(0)
        i = 1 #iteration count
        # I started this outside the loop so I could iterate on L1
        pi_to_k = startVector
        pi_to_k_plus_one = (0.85 * (mtx.dot(pi_to_k))) + (0.85 * (danglingNodeVector.dot(pi_to_k) + 0.15) * articleVector)

        l1 = np.linalg.norm(pi_to_k_plus_one - pi_to_k)
        print("L1: {} on Iteration {}".format(l1, i))

        while l1 >= self.const_epsilon:
            i += 1
            
            pi_to_k = pi_to_k_plus_one
            pi_to_k_plus_one = (0.85 * (mtx.dot(pi_to_k))) + (.85 * (danglingNodeVector.dot(pi_to_k) + 0.15) * articleVector)          
            l1 = np.linalg.norm((pi_to_k_plus_one - pi_to_k))
           
            print("L1: {} on Iteration {}".format(l1, i))
        print("Barely Made it!  Shame about Johnson")
        return pi_to_k_plus_one
    def findEigenFactor(self, mtx, influenceVector):
        print("Calculating Eigenfactor.  Please keep your arms and legs inside the vehicle at all times.")
        #mtx = mtx.fillna(0)
        ef = 100 * ((mtx.dot(influenceVector)) / (mtx.dot(influenceVector).sum()))

        print(ef)
        print("And now we know the cost of not listening to your tour guide: gross limb removal.")
        return ef


    #Data Ingestion
    def Load_File(self, filepathstring):
        print("Loading File")
        loaded_file = []
        with open(filepathstring) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for row in csv_reader:
                loaded_file.append(row)
        return loaded_file







#myEFF = EigenFactorFinder.test_Matrix()
myEFF = EigenFactorFinder()
        

# My Answers:
# a)  Top 20:
# 8929  1.108543
# 724   0.247855
# 238   0.244281
# 6522  0.235651
# 6568  0.226109
# 6696  0.225744
# 6666  0.216824
# 4407  0.206529
# 1993  0.201796
# 2991  0.184877
# 5965  0.182961
# 6178  0.180931
# 1921  0.175256
# 7579  0.170833
# 899   0.170400
# 1558  0.168341
# 1382  0.163671
# 1222  0.150992
# 421   0.149528
# 5001  0.149351
#
# b) Time Taken: 23.157s on the last execution
# c) Number of iterations: L1: 9.282858497531677e-06 on Iteration 26