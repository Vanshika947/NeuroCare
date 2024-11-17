import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv('/content/sample_data/parkinsons_datset.csv')

df.head()

df.tail()

df.shape

df=df.drop(columns='name', axis=1)
df.head()

df.shape

df.info()

df.describe()

df.isnull().sum()

df['status'].value_counts()

df.groupby('status').mean()

df['status'].value_counts().plot(kind="bar")
plt.xlabel("Distinct Values")
plt.ylabel("Number of Occurrences")
plt.show()

df.hist(figsize = (12,14))
plt.show()


plt.rcParams['figure.figsize'] = (15, 4)
sns.pairplot(df,hue = 'status', vars = ['MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ', 'Jitter:DDP'] )
plt.show()

plt.rcParams['figure.figsize'] = (15, 4)
sns.pairplot(df,hue = 'status', vars = ['MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA'] )
plt.show()


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True,cmap ='Blues')
plt.show()

X=df.drop(columns='status',axis=1)
Y=df['status']

print(X)

print(Y)

scaler= StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
print(standardized_data)

X = standardized_data
Y= df['status']

print("X is: ",X)
print("*************************************************************************")
print("Y is: ",Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

print(Y.shape, Y_train.shape, Y_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

train_prediction= classifier.predict(X_train)
train_accuracy= accuracy_score(train_prediction, Y_train)
print("Accuracy score on the training data is: ",train_accuracy)

test_prediction= classifier.predict(X_test)
test_accuracy= accuracy_score(test_prediction, Y_test)
print("Accuracy score on the test data is: ",test_accuracy)

print(confusion_matrix(test_prediction, Y_test))
print(classification_report(test_prediction, Y_test))

input_data= (199.228,209.512,192.091,0.00241,0.00001,0.00134,0.00138,0.00402,0.01015,0.089,0.00504,0.00641,0.00762,0.01513,0.00167,30.94,0.432439,0.742055,-7.682587,0.173319,2.103106,0.068501)

input_data_nparray =np.asarray(input_data)

input_data_reshaped =input_data_nparray.reshape(1,-1)

std_data =scaler.transform(input_data_reshaped)

prediction=classifier.predict(std_data)

if (prediction[0]==1):
    print("The person has Parkinson's")
else:
    print("The person doesn't have Parkinson's")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv('/content/sample_data/parkinsons_datset.csv')

X = df.drop(['name', 'status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
