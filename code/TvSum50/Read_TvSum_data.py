

from pandas import DataFrame
import numpy as np
df = DataFrame.from_csv("../../Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno_cols.tsv", sep="\t")
#df.columns = ['Category','Importance Scores']

def save_targets():
    for i in xrange(0, df.shape[0], 20): 
        numbers = df['Scores'][i].split(',')
        numbers = [int(x) for x in numbers]
        average = np.zeros((1, len(numbers)))
#        print('average = ', average.shape)
#        print('nmumsber = ', len(numbers))
        for j in range(20):
            single_user = []
            videoName = df.index[i+j]
            numbers = df['Scores'][i+j].split(',')
            category = df['Category'][i+j]
            
            numbers = [int(x) for x in numbers]
            numbers = np.array(numbers)
            numbers = np.expand_dims(numbers, axis=0)
#            print('averagedsadsa = ', average.shape)
#            print('nmumsberadsdsa = ', len(numbers))
            average = average + numbers
        
            if (j % 19 == 0) and j!=0:
#                print('here')
                average = average / 100
#                print(average)
                np.save('../../saved_numpy_arrays/TvSum50/average_targets/' + df.index[i+j] + ' ' + category  + '.npy', average)
#                return average[0]
                
            

#aver = save_targets()