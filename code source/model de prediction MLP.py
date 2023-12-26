from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/dataset.csv')

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df["Gender"]=le.fit_transform(df['Gender'])
df["Geography"]=le.fit_transform(df['Geography'])
df["Card Type"]=le.fit_transform(df['Card Type'])

df= df.drop(["RowNumber" , "CustomerId" , "Surname"] , axis=1)

import pandas as pd
import numpy as np
seuil_Balance = 5000
seuil_Tenure = 5
seuil_NumOfProducts = 3
Age = 30
Seuil_CreditScore = 3000

def calculer_score_fidelite(row):
    score = 0
    if row["Balance"] > np.quantile(row["Balance"], .80):
        score += 3

    if row['CreditScore'] > np.quantile(row["CreditScore"], .80):
        score += 3

    if row['Tenure'] > seuil_Tenure:
        score += 3

    if row['NumOfProducts'] > seuil_NumOfProducts:
        score += 3

    if row["HasCrCard"] ==1:
        score +=3

    if row["IsActiveMember"] ==1:
        score +=3

    if row["Complain"] <= 2:
        score +=3

    if row['Satisfaction Score'] >= 3:
        score += 3

    if row["Point Earned"] >100:
        score += 3
    if score <= 15 :
      score=0
    else :
        score=1

    return score

df['fidéle'] = df.apply(calculer_score_fidelite, axis=1)

###############################################

df.to_csv('/content/drive/MyDrive/fidélisation.csv', index=False)

df.info()
labels=df['fidéle'].value_counts()
labels
df.isnull().sum()
###############################################

import matplotlib.pyplot as plt
import seaborn as sns
numerical_column= ["CreditScore","Age","Tenure","Balance","EstimatedSalary","NumOfProducts","Satisfaction Score" , "Point Earned"]
plt.figure(figsize=(20,15))

for i,j in zip(range(1, 9),numerical_column):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, x=j)
    sns.set_theme()
    plt.title('Boxplot of {}'.format(j))

col = ["CreditScore" ]
Q1=df[col].quantile(0.25)
Q3=df[col].quantile(0.75)
IQR = Q3 -Q1
df = df[(~((df[col] < (Q1 - 1.5 * IQR)))).any(axis=1)]

df.shape

df.head()

matrix = df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(matrix, annot=True)
plt.show()

df=df.drop('Exited' , axis=1)

import numpy as np

variance = np.var(df)
variance
df.shape
df.head()

from sklearn.feature_selection  import VarianceThreshold , SelectKBest , chi2
sel = VarianceThreshold(threshold=(0.20))
sel.fit_transform(df)
sel.get_support()

df.shape
df.head()

df = df.dropna()

from imblearn.over_sampling import SMOTE

X=df.iloc[:, :-1]
y=df.iloc[: , -1]

sm = SMOTE(random_state=42)

X, y = sm.fit_resample(X, y)

y.value_counts()

from sklearn.ensemble import ExtraTreesClassifier

Model =ExtraTreesClassifier()
Model.fit(X , y)
print(Model.feature_importances_)

importances =Model.feature_importances_
std = np.std([Model.feature_importances_ for tree in Model.estimators_] , axis =0 )
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature important")
plt.bar(range(X.shape[1]) , importances[indices] , color= "r" , align="center")
plt.xticks(range(X.shape[1]) , indices)
plt.xlim([-1 , X.shape[1]])
plt.show()

X = SelectKBest(chi2 , k=10).fit_transform(X,y)
X.shape


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.4, random_state=42)
validation_data, X_test, validation_labels, Y_test = train_test_split(X_test, Y_test , test_size=0.5, random_state=42)

X_train.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
validation_data = scaler.fit_transform(validation_data)

from sklearn import preprocessing
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)
validation_data = preprocessing.normalize(validation_data)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(8,8),
                    random_state=2,
                    verbose=True,
                    learning_rate_init=0.001
                   )
clf.fit(X_train , Y_train)

ypredict = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test , ypredict)

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_true=Y_test, y_pred=ypredict)
print(matrix)
sns.heatmap(matrix, annot=True)

import pickle

model_pkl_file = "/content/drive/MyDrive/mlp_classifier.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(clf, file)

from sklearn.metrics import classification_report

with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)

# evaluate model 
y_predict = model.predict(X_test)

# check results
print(classification_report(Y_test, y_predict))