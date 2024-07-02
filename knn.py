from ucimlrepo import fetch_ucirepo 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold #using k-fold cross-validation to train model
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


if __name__ == '__main__':
  
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
  
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    X_clean = X.dropna()
    y_clean = y.dropna()

    best_neighbor_value, max_score = 0, 0

    #store the predictions and test values of the best model so that we can do a confusion matrix later
    confusion_pred, confusion_test = [], []

    #initializing y_pred and y_test to avoid edge case errors, these will store the predictions of the model and the actual values, respectively
    y_pred = []
    y_test = []

    #number of folds we will use for k-fold cross-validation, after researching it looked like 5 and 10 were the common values to use
    num_folds = 5

    #looping through different values of n_neighbors to make the models and get their scores, looking at what a good number of neighbors is without overfitting made me come to the conclusion that sqrt(how many datapoints there are) was considered a default start
    for i in range(2, round(math.sqrt(len(X_clean.index)))): 
        try:
            model = KNeighborsClassifier(n_neighbors = i, weights='distance')
            kfold = KFold(n_splits = num_folds, shuffle = True) 

            score = 0.0

            #going through each of the k folds for k-fold cross-validation
            for train_index, test_index in kfold.split(X_clean):
                X_train = X_clean.iloc[train_index]
                y_train = y_clean.iloc[train_index]
                X_test = X_clean.iloc[test_index]
                y_test = y_clean.iloc[test_index]
                model.fit(X_train, y_train.values.ravel())
                y_pred = model.predict(X_test)
                score += accuracy_score(y_pred, y_test)

            score = score / num_folds
            if score > max_score:
                max_score = score
                best_neighbor_value = i
                confusion_pred = y_pred
                confusion_test = y_test
        except:
            break

    print(max_score, best_neighbor_value, num_folds)
    
    confusion = confusion_matrix(confusion_test, confusion_pred)
    val = np.asmatrix(confusion)
    classnames = list(set(confusion_pred))
    df_cm = pd.DataFrame(val)
    plt.figure()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('KNN Model Results')
    plt.savefig('knn_confusion_matrix.png')
    plt.show()
