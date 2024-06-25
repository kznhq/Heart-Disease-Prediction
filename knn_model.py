from ucimlrepo import fetch_ucirepo 
# from sklearn.neighbors import KNeighborClassifier  

# fetch dataset
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 


print(X.head())
print(y.head())
