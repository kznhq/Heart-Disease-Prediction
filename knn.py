from ucimlrepo import fetch_ucirepo 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold #using k-fold cross-validation to train model

if __name__ == '__main__':
  
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
  
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    X_clean = X.dropna()
    y_clean = y.dropna()

    best_neighbor_value, max_score = 0, 0

    #number of folds we will use for k-fold cross-validation, after researching it looked like 5 and 10 were the common values to use
    num_folds = 15

    num_neighbors = 2

    #looping through different values of n_neighbors to make the models and get their scores
    while True:
        try:
            model = KNeighborsClassifier(n_neighbors = num_neighbors)
            kfold = KFold(n_splits = num_folds) 

            score = 0.0

            #going through each of the k folds for k-fold cross-validation
            for train_index, test_index in kfold.split(X_clean):
                X_train = X_clean.iloc[train_index]
                y_train = y_clean.iloc[train_index]
                X_test = X_clean.iloc[test_index]
                y_test = y_clean.iloc[test_index]
                model.fit(X_train, y_train.values.ravel())
                score += model.score(X_test, y_test)

            score = score / num_folds
            if score > max_score:
                max_score = score
                best_neighbor_value = num_neighbors
            num_neighbors += 1
            if num_neighbors == 200:
                print('30')
                print(score)
        except:
            break

    print(max_score, best_neighbor_value, num_folds)
