import pandas as pd
import pickle
disease=pd.read_csv('training disease.csv')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(disease.loc[:,'itching':'yellow_crust_ooze'],disease['prognosis'])

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pickle.dump(knn,open('model.pk1','wb'))
model=pickle.load(open('model.pk1','rb'))
print(model.predict(X_test))
