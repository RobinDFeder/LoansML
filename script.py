import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# load the data to pnd
loan_dataset = pd.read_csv('/content/dataset.csv')

type(loan_dataset)

loan_dataset.head()

#how big the dataset is
loan_dataset.shape

#stat measure
loan_dataset.describe()

# how many nan there are 
loan_dataset.isnull().sum()

# drop the missing value
loan_dataset = loan_dataset.dropna()

loan_dataset.isnull().sum()

#encoding labels
loan_dataset.replace({"Loan_Status":{"N":0, "Y":1}},inplace=True)

loan_dataset.head()

# dependents 
loan_dataset['Dependents'].value_counts()

# replace 3+ to 4 to have a hard valuee
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# dependents 
loan_dataset['Dependents'].value_counts()

#visualization of the data 
#education & loans already active
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

#visual for marital & loan 
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)

# converting labels values to num values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

loan_dataset.head()

# sorting data out
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

print(X)
print(Y)

# The big Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#Train to become better
#Vector Machine Model
classifier = svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)

#Model Evaluation
#accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

classifier.fit(X_train,Y_train)

#accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

#accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print('Accuracy on test data : ', test_data_accuracy)