import numpy as np
import operator
from collections import  Counter
import seaborn as sns

def Data_Preperation():

    #Preparing Data

    mu=0
    sigma=1
    samples1=100
    class1=np.random.normal(mu,sigma,100)   #drawing samples from gaussian distribution

    samples2=80
    low=0
    high=1
    class2=np.random.uniform(low,high,samples2) #drawing samples from uniform distribution

    data=np.concatenate((class1,class2),axis=0)
    data = data.reshape(data.shape[0],1)
    label=np.empty(shape=(180,1))                    #Classes for training data
    label[0:samples1,:]=1
    label[samples1+1:samples1+samples2,:]=2

    return data,label

def KNN(data,label):

    #split data into training and testing

    # Split dataset 80:20 Training and 90 testing
    samples=180
    #train_split=int(samples*0.8)
    #test_split=int(samples*0.2)
    #train = data[0:train_split, :]  # First 90 rows
    #test = data[train_split:samples,]  # Last 90 rows
    #print train.shape
    #print test.shape
    #exit(0)

    n_neighbors=11
    Euclidean = []
    Neighbors = []
    indices=[]
    class_list=[]
    y_pred_list=[]
    for x in range(0, samples):
        for y in range(0,samples):

            if x!=y:
                dist = (data[y],np.linalg.norm(data[x, :] - data[y,:]))  #compute euclidean distance for each point in train set
                Euclidean.append(dist)
                Euclidean.sort(key=operator.itemgetter(1)) #sorting in ascending order

        for i in range(n_neighbors):
            Neighbors.append(Euclidean[i][0])  #11 neighbors closest to x in training data


        # find most common class among 11 nearest neighbors

        data_list=data.tolist()
        for i in Neighbors:  #retrieve index
                index=data_list.index(i)
                indices.append(index)

        #Now do voting for majority class for classification
        label_list=label.tolist()
        for i in indices:
            class_list.append(label_list[i])
            flat_class_list = [x for sublist in class_list for x in sublist]
            majority_class=Counter(flat_class_list)
            y_pred_list.append(majority_class.most_common(1)[0][0])

    
        #Re initializing all the lists for other data points in the dataset

        Euclidean=[]
        Neighbors=[]
        indices=[]
        class_list=[]

    # Now compute accuracy
    # origional labels are in "labels"
    #precited labels are in "y_pred"

    count=0
    label = label.tolist()
    for i in range(0,len(data)):
        datapoint=data[i]
        y_true=label[i]
        y_pred = y_pred_list[i]
        if y_true[0]==y_pred:
            count+=1
        else:
            continue

    print "Accuracy",str(float(count/float(len(data)))*100)+"%"


def Main():

    data,label=Data_Preperation()
    KNN(data,label)


Main()