# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ameen Basha A
RegisterNumber:  212224220008
*/
```
```

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report

data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)

data1.isnull().sum()
data1.duplicated().sum()

le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_p"]=le.fit_transform(data1["ssc_p"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["status"]=le.fit_transform(data1["status"])

data1.head()

x=data1.iloc[:,:-1]
y=data1["status"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)

cr=classification_report(y_test,y_pred)
print("Classification Report:\n",cr)



```

## Output:
![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
