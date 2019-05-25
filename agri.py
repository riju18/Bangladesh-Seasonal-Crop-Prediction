import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,cross_val_score

df = pd.read_csv('C:/Users/This Pc/Desktop/data_science_begin/undergraduate_thesis/agri_cow dung/agri1.1.csv', index_col = 0)

df['Poultry_Manure'] = df['Poultry_Manure'].fillna(0.0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Area'] = le.fit_transform(df['Area'])
df['PH'] = le.fit_transform(df['PH'])
df['Season'] = le.fit_transform(df['Season'])
df['Crop'] = le.fit_transform(df['Crop'])

X = df.iloc[:, :-2].values
y = df.iloc[:, -2].values
y1= df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test) 

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train, y_train)
y_predict = lg.predict(x_test)

print(accuracy_score(y_test, y_predict) * 100)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
print(accuracy_score(y_test, svc_pred) * 100)
print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))