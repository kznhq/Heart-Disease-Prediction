from ucimlrepo import fetch_ucirepo 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold #using k-fold cross-validation to train model
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 

    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    X_clean = X.dropna()
    y_clean = y.dropna()

    max_score = 0
    confusion_pred = []
    confusion_test = []
    y_pred = []
    y_test = []
    num_features = 0

    num_folds = 10

    #looping through different values of n_neighbors to make the models and get their scores, looking at what a good number of neighbors is without overfitting made me come to the conclusion that sqrt(how many datapoints there are) was considered a default start
    for i in range(2, (len(X_clean.index))): 
        try:
            # model = DecisionTreeClassifier(min_samples_leaf=i, criterion='entropy')
            model = DecisionTreeClassifier(min_samples_leaf = 5)
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
                confusion_pred = y_pred
                confusion_test = y_test
        except:
            break

    print(max_score, num_folds)
    
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
    plt.title('DT Model Results with low min_samples_leaf = 5')
    plt.savefig('dt_confusion_matrix_low_msl.png')
    plt.show()
