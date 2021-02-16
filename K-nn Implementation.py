import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_predict


data=pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]


data.diagnosis=[ 1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values 
x_data=data.drop(["diagnosis"],axis=1)


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)


print("KNN score : %",knn.score(x_test,y_test)*100)
print("Tolerance : %",100-knn.score(x_test,y_test)*100)



score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))


plt.plot(range(1,15),score_list)
plt.xlabel("K values")
plt.ylabel("Accuracy Rate")
plt.show()



cross_predict = cross_val_predict(knn2,x_test,y_test,cv=10)
print(metrics.confusion_matrix(y_test, cross_predict))
print(metrics.classification_report(y_test, cross_predict))
