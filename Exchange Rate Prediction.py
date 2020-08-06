#Importing the libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold

#Getting the data and setting the index in date time format

dataset = pd.read_csv("Classification GITHUB data.csv")
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.set_index('Date')


#Creating the additional differential columns

data = dataset.iloc[1826:, 0:]
data['Interest Differential'] = data['Turkey Interest Rate'] - data['US Interest Rate']
data['Turkey Public Debt '] = data['Turkey Public Debt ']/1000
data['Turkey Public Debt'] = data['Turkey Public Debt ']
data['Growth Rate Differential'] = data['Turkey GDP Growth Rate'] - data['US GDP Growth Rate']


#Picking out the relevant columns

X = data.iloc[:, [5, 6, 10, 13, 14, 15]].values
y = data.iloc[:, [0]].values


#Function to average column values over given intervals

def chunk_it_up(array, n):
    
    size = np.shape(array)
    rows = size[0]
    columns = size[1]
    
    largest = int(rows/n)
    remainder = rows%n
    
    avg_matrix = np.zeros([largest, columns])
    
    for i in range(largest):
        for j in range(columns):
             avg_matrix[i][j] = np.mean(array[(i*n):(i*n+n), j])
             
    rem_matrix = []
    
    if remainder > 0:
        for j in range(columns):
            rem_matrix.append(np.mean(array[(largest*n):, j]))
            
        rem_matrix = np.reshape(rem_matrix, [1, len(rem_matrix)])
    
        avg_matrix = np.append(avg_matrix, rem_matrix, axis = 0)
        
    
    return avg_matrix


#Taking the weekly averages

X_5 = chunk_it_up(X, 5)
y_5 = chunk_it_up(y, 5)


#Standardising variables is important here for support vector regression

std = np.std(y_7)
mean = np.mean(y_7)

X_7_std = (X_7 - np.mean(X_7))/np.std(X_7)
y_7_std = (y_7 - mean)/std


#Roll it up one place

y_7_std_2 = np.roll(y_7_std, -1)
y_7_2 = np.roll(y_7, -1)


#Splitting into the training and test sets for the weekly average analysis

X_train_1 = X_7_std[0:-2, :]
X_test_1 = X_7_std[-2:-1, :]
y_train_1 = y_7_std_2[0:-2, :]
y_test_1 = y_7_2[-2:-1, :]


X_train_2 = X_7_std[0:-3, :]
X_test_2 = X_7_std[-3:-1, :]
y_train_2 = y_7_std_2[0:-3:, :]
y_test_2 = y_7_2[-3:-1, :]


X_train_4 = X_7_std[0:-5, :]
X_test_4 = X_7_std[-5:-1, :]
y_train_4 = y_7_std_2[0:-5:, :]
y_test_4 = y_7_2[-5:-1, :]


X_train_12 = X_7_std[0:-13, :]
X_test_12 = X_7_std[-13:-1, :]
y_train_12 = y_7_std_2[0:-13:, :]
y_test_12 = y_7_2[-13:-1, :]



#Defining some output functions

#Function for calculating the MAPE

def get_my_mape(array_pred, array_real):
    mid_1 = []
    for i in range(len(array_pred)):
        mid_1.append((array_pred[i] - array_real[i])/array_real[i]*100)
    mid_1_abs = [np.abs(mid_1[i]) for i in range(len(mid_1))]
    output = sum(mid_1_abs)/len(mid_1)
    return output

#Function for performing weekly average analysis

def lets_evaluate(classifier):
    classifier.fit(X_train_1, y_train_1)
    pred_for_1 = classifier.predict(X_test_1)
    pred_for_1_og = pred_for_1*std + mean
    mape_for_1 = get_my_mape(pred_for_1_og, y_test_1)
    
    classifier.fit(X_train_2, y_train_2)
    pred_for_2 = classifier.predict(X_test_2)
    pred_for_2_og = pred_for_2*std + mean
    mape_for_2 = get_my_mape(pred_for_2_og, y_test_2)
    
    classifier.fit(X_train_4, y_train_4)
    pred_for_4 = classifier.predict(X_test_4)
    pred_for_4_og = pred_for_4*std + mean
    mape_for_4 = get_my_mape(pred_for_4_og, y_test_4)
    
    classifier.fit(X_train_12, y_train_12)
    pred_for_12 = classifier.predict(X_test_12)
    pred_for_12_og = pred_for_12*std + mean
    mape_for_12 = get_my_mape(pred_for_12_og, y_test_12)
    
    
    output_dict = {
            'MAPE for last 1 week prediction': mape_for_1,
            'MAPE for last 2 week prediction': mape_for_2,
            'MAPE for last 4 week prediction': mape_for_4,
            'MAPE for last 12 week prediction': mape_for_12,
            }
    
    return output_dict



#Testing out various regression models

regressor_rbf = SVR(kernel = 'rbf', gamma = 'auto')
matrix_rbf = lets_evaluate(regressor_rbf)

regressor_poly_2 = SVR(kernel = 'poly', degree = 2, gamma = 'auto')
matrix_poly_2 = lets_evaluate(regressor_poly_2)

regressor_poly_3 = SVR(kernel = 'poly', degree = 3, gamma = 'auto')
matrix_poly_3= lets_evaluate(regressor_poly_3)

regressor_linear = SVR(kernel = 'linear', gamma = 'auto')
matrix_linear = lets_evaluate(regressor_linear)


#Grid searching for the best C value for the RBF kernel

param_grid = {"C": [0.1, 1, 3, 5, 10]}

reg = GridSearchCV(estimator = regressor_rbf, param_grid = param_grid, cv=KFold(), scoring = 'neg_mean_squared_error')

grid_search_results = reg.fit(X_7_std[0:-1, :], y_7_std_2[0:-1, :])

grid_search_results.best_score_
grid_search_results.best_params_


#Grid searching for the best epsilon value

param_grid_2 = {
        "C": [1],
        "epsilon": [0.01, 0.1, 1, 3, 5, 10]
        }

reg_2 = GridSearchCV(estimator = regressor_rbf, param_grid = param_grid_2, cv=KFold(), scoring = 'neg_mean_squared_error')

grid_search_results_2 = reg_2.fit(X_7_std[0:-1, :], y_7_std_2[0:-1, :])

grid_search_results_2.best_score_
grid_search_results_2.best_params_


#Evaluating the final regressor with the tuned hyperparameters

final_regressor = SVR(kernel = 'rbf', gamma = 'auto', C = 1.0, epsilon = 0.01)
final_regressor.fit(X_7_std, y_7_std_2)
matrix_final = lets_evaluate(final_regressor)


#Defining them like this for convenience

y_7_std_2 = y_7_std_2[0:-1, :]
X_7_std = X_7_std[0:-1, :]


#Performing K-Fold cross-validation to check for the most robust model

scores_rbf = sum(cross_val_score(regressor_rbf, X_7_std, y_7_std_2, cv=83, scoring = 'neg_mean_squared_error'))/83
scores_poly_2 = sum(cross_val_score(regressor_poly_2, X_7_std, y_7_std_2, cv=83, scoring = 'neg_mean_squared_error'))/83
scores_poly_3 = sum(cross_val_score(regressor_poly_3, X_7_std, y_7_std_2, cv=83, scoring = 'neg_mean_squared_error'))/83
scores_linear = sum(cross_val_score(regressor_linear, X_7_std, y_7_std_2, cv=83, scoring = 'neg_mean_squared_error'))/83
scores_final = sum(cross_val_score(final_regressor, X_7_std, y_7_std_2, cv=83, scoring = 'neg_mean_squared_error'))/83




