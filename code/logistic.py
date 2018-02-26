# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:05:10 2018

code ref:https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
also resusing code from week 10 coureswork
@author: wangxinji
"""
from random import randrange
from csv import reader
from math import exp
import os 
import time
import pandas as pd
from math import sqrt

os.chdir('C:/Users/Hasee/Desktop')
# Load dataset
def load_data(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			dataset.append(row)
	return dataset

data = load_data('pima-indians-diabetes.csv')

    
# turn dataset in to float
#because we are loading dataset into a list, we should change the class to data from string to float
def chfloat(dataset, column):
    for row in dataset:
        if isinstance(row[column],float) == False:
	        row[column] = float(row[column].strip())
        else:
            pass
 
# scale dataset,search each maximum and minimum value in each row and then scale the dataset 
def scale(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# crossvalidation            

def crossvalidation(dataset,n_folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)   #copy the dataset 
    fold_size = int(len(dataset)/n_folds)  #calculate fold size so that we can decide how much dataset in a split
    for i in range(n_folds):
        fold = list()
        while len(fold)<fold_size:
            index = randrange(len(dataset_copy)) # randomly pick items in the dataset,copy it in the fold data 
            fold.append(dataset_copy.pop(index))#then use delete the pop-out data from the dataset
        dataset_split.append(fold)
    return dataset_split

#get accuracy
def get_acc(y_truth,y_predict):
    match = 0  # match count start from zero,while each match the count increse by one 
    for i in range(len(y_truth)):
        if y_predict[i] == y_truth[i]:
            match +=1
        else:
            pass
    acc = match/len(y_truth)
    return acc 

def predict(row,coef):
    #in the logistic regression we use sigmoid function
    yhat = coef[0]

    for i in range(len(row)-1):	
        yhat += coef[i + 1] * row[i]  #we could implement this step by using vectorize our dataset to decrese computation time
    return 1.0 / (1.0 + exp(-yhat))
    
def sgd(train,learning_rate,n_epoch):
    coef = [0 for i in range(len(train[0]))]   #initilize weight(coef)
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row,coef) #in each epch calculate yhat and error , renew weights and intercepts after each calculation
            error =  row[-1]-yhat
            coef[0] = coef[0] + learning_rate * error * yhat * (1 - yhat) #this is the intercept,b
            sum_error += error**2
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] + learning_rate * error *yhat * (1 - yhat) * row[i] #this is the weights
    return coef 


def logistic(train,test,learning_rate = 0.1,n_epoch = 300,threshold = 0.5):
    prediction = list()
    coef = sgd(train,learning_rate,n_epoch)
    for row in test:
        yhat = predict(row,coef)
        if yhat >= threshold: #we can modify this probability to get better result 
            yhat = 1
        else:
            yhat = 0
        prediction.append(yhat)
    return prediction
    
def evaluate(dataset,algorithm = logistic,n_folds = 10 ,*args):#the args here is set to receive other parametres
    folds = crossvalidation(dataset)
    scores = list()
    count = 1
    std = 0
    for fold in folds:  
        #we have 10 folds in default,in each fold train and get a model accuracy   
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set,[])
        test_set = list()
        for row in fold:
            _row = list(row)
            test_set.append(_row)
            _row[-1]= None
        predicted = algorithm(train_set,test_set,*args)
        actual = [row[-1] for row in fold]
        accuracy = get_acc(actual,predicted)
        scores.append(accuracy)
        std = sum(scores)
        print (count,'fold acc is ',accuracy)
        count +=1
    print('your std error=  ' ,str(std/(sqrt(768))))   #76 is the objects in one fold(under 10 fold-crossvalidation)
    return scores,predicted


#following function calculates the performance metrics , we focus on recall and precision,f1-score
def get_performance(predicted,dataset):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(predicted)):
        if predicted[i] == 1 and  dataset[i][-1] ==1:
            TP += 1
        if predicted[i] == 1 and  dataset[i][-1] ==0:
            FP += 1
        if predicted[i] == 0 and  dataset[i][-1] ==0:
            TN += 1
        if predicted[i] == 0 and  dataset[i][-1] ==1:
            FN += 1
    print('TP = ',TP, 
          'FP = ',FP,
          'TN = ' ,TN,
          'FN = ' ,FN
          )
    if TP + FP == 0:
        pass
    else:
        precision = TP/(TP+FP)
        print('precision = ' ,precision)
        if TP + FN == 0:
            pass
        else:
            recall = TP/(TP+FN)
            print ('recall = ',recall)
            print('your model f1-score is ' ,str(2*precision*recall/(precision+recall)))


def multiacc(dataset,predict):
    match =0
    for i in range(len(predict)):
        if float(dataset[i][-1]) == predict[i]:
            match = match +1
    acc = match/len(predict)
    return acc 
# get performance metrics we have to compare the ture value and the predict value
def pred(dataset,coef,threshold):
    prediction= list()
    y_pred = list()
    for row in dataset:
        yhat = predict(row, coef)
        prediction.append(yhat)
        #we set the treshold 0.5 but indeed it could be adjust inorder to fit reality
        if yhat >=threshold:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
    return  prediction,y_pred



''' 
test case code for the pima-indian-diabetes dataset(binary)
'''
time_record =list()

#time comsuming calculation

tic = time.time()        
# data
filename = 'pima_pca.csv'
dataset = load_data(filename)
for i in range(len(dataset[0])):
	chfloat(dataset, i)
# normalize
scale(dataset)

# evaluate algorithm
n_folds = 10
learning_rate = 0.15
n_epoch = 300
threshold =0.8

scores,predicted = evaluate(dataset, logistic, n_folds, learning_rate, n_epoch,threshold)
print('Mean Accuracy:' , (sum(scores)/float(len(scores))))
toc = time.time()
print('runtime : ',str(toc-tic),'s')
time_record.append(toc-tic)


#to get prediction
coef = sgd(dataset,learning_rate,n_epoch)
prediction,y_pred = pred(dataset,coef,threshold)
#performance matrix and calculate metircs
get_performance(y_pred,dataset)


'''
                  test case code for the sonar dataset(binary)
in this dataset :
                M for Mine ,and is transform into 0
                R for Rock, and is transform into 1

'''
#time comsuming calculation
tic = time.time()        
# load data
filename = 'sonar_pca.csv'
dataset = load_data(filename)
for i in range(len(dataset[0])):
	chfloat(dataset, i)
# normalize
scale(dataset)

# evaluate algorithm
n_folds = 10
learning_rate = 0.25
n_epoch =400
threshold = 0.8

scores,predicted = evaluate(dataset, logistic, n_folds, learning_rate, n_epoch,threshold)
print('Mean Accuracy:' , (sum(scores)/float(len(scores))))
toc = time.time()

#to get prediction
coef = sgd(dataset,learning_rate,n_epoch)
prediction,y_pred = pred(dataset,coef,threshold)
#performance matrix and calculate metircs
get_performance(y_pred,dataset)
print('runtime : ',str(toc-tic),'s')



'''
time record test case
'''
time_record =list()
#run 20times to record time for t test
for i in range(20):
    tic = time.time()        
    # data
    #filename = 'pima_pca.csv'
    filename = 'pima-indians-diabetes.csv'
    dataset = load_data(filename)
    for i in range(len(dataset[0])):
    	chfloat(dataset, i)
    # normalize
    scale(dataset)
    
    # evaluate algorithm
    n_folds = 10
    learning_rate = 0.15
    n_epoch = 300
    threshold =0.8
    
    scores,predicted = evaluate(dataset, logistic, n_folds, learning_rate, n_epoch,threshold)
    print('Mean Accuracy:' , (sum(scores)/float(len(scores))))
    toc = time.time()
    print('runtime : ',str(toc-tic),'s')
    time_record.append(toc-tic)
    
pca_time = pd.DataFrame({'PCA':time_record})
#pima_time = pd.DataFrame({'PCA':time_record})

time_result = pd.concat([pima_time,pca_time], axis=1) #copy the result into a csv file then analyze using R


''' 
one vs all logistic regression
test case code for the multiclass dataset--- wine
'''
filename = 'wine.csv'
wine = load_data(filename)  #we only need the label column here-->to compare the label and calculate acc

# to realize one vs all, we adjust the dataset first
#time comsuming calculation
tic = time.time()        
# evaluate algorithm
n_folds = 10
learning_rate = 0.1
n_epoch =400

filename = 'wine1.csv'
wine1 = load_data(filename)

for i in range(len(wine1[0])):
	chfloat(wine1, i)
# normalize
scale(wine1)
coef_wine1 = sgd(wine1,learning_rate,n_epoch)

filename = 'wine2.csv'
wine2 = load_data(filename)

for i in range(len(wine2[0])):
	chfloat(wine2, i)
# normalize
scale(wine2)
coef_wine2 = sgd(wine2,learning_rate,n_epoch)


filename = 'wine3.csv'
wine3 = load_data(filename)

for i in range(len(wine3[0])):
	chfloat(wine3, i)
# normalize
scale(wine3)
coef_wine3 = sgd(wine3,learning_rate,n_epoch)

#get predicted probability
prediction1,y_pred1 = pred(wine1,coef_wine1)
prediction2,y_pred2 = pred(wine2,coef_wine2)
prediction3,y_pred3 = pred(wine3,coef_wine3)

df1 = pd.DataFrame({'wine1':prediction1})
df2 = pd.DataFrame({'wine2':prediction2})
df3 = pd.DataFrame({'wine3':prediction3})

get_performance(y_pred1,wine1)
get_performance(y_pred2,wine2)
get_performance(y_pred3,wine3)

result = pd.concat([df1, df2,df3], axis=1)
#get the maximum value of each line  (code reuse from the data mining group project  line356~line 360)
predicted_numbers = []
for i in range(result.shape[0]):
    to_assess = list(result.iloc[i])
    predicted_numbers.append(to_assess.index(max(to_assess)))

pn = [i + 1 for i in predicted_numbers]  #python start with 0 , but label start with 1 so add the '1' back

multiacc(wine,pn)
toc = time.time()        
print('runtime = ',str(toc-tic),'s')
