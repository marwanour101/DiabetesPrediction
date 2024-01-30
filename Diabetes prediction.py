import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set()
#Reading data
df=pd.read_csv("F:\\Semester_2\\AI\\diabetes_binary_health_indicators_BRFSS2015.csv")
#pre-processing
print(df.head())
print("summution of null",df.isnull().sum())
df = df.dropna()
print(df.shape)
df=df.drop_duplicates()
print(df.shape)
print(df.describe().round())
print(df.info())
#corelation data
coor=df.corr()
print(sns.heatmap(coor,annot=True).set_title("corelation data"))
plt.show()
# show dataframe before balance 
print(sns.countplot(x='Diabetes_binary',data=df).set_title("0 -> non_diabetic   1-> diabetic"))
plt.show()

# min max scaller data scaling  
scaling=MinMaxScaler()
df=pd.DataFrame(scaling.fit_transform(df),columns=df.columns)
#balancing data use oversample
x=df.iloc[:,1:22]
print(x.shape)
y=df['Diabetes_binary']
counter = Counter(y)
print(counter)
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
#feature selection
#feature=SelectKBest(score_func=chi2,k=10)
model=LinearRegression()
feature=SelectFromModel(estimator=model)
x=feature.fit_transform(x,y)
print(x.shape)
print(feature.get_support())
#tranining and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=0)
print("---------shapes for feature and target----------- ")
print(x.shape)
print(y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#data scaling
scaling=StandardScaler(copy=True,with_mean=True,with_std=True)
scaling.fit(X=x)
standardizd=scaling.transform(X=x)
print(standardizd)
print(standardizd.shape)
#after scaling featura and target
X_decision=standardizd
y_decision=y
#SVM
clf= svm.LinearSVC()
clf.fit(X_train,y_train)
y_pred1= clf.predict(X_test)
print("-----------------------------------------------------SVM--------------------------------------------------------------------- ")
print(y_pred1)
print("\n train score is : {:.3f}".format(clf.score(X=X_train, y=y_train)))
print("test score is : {:.3f}".format(clf.score(X=X_test, y=y_test)))
print(clf.score(X_train, y_train))
print(classification_report(y_test, y_pred1))
cf_matrix = confusion_matrix(y_test, y_pred1)
print("confusion_matrix :",cf_matrix)
print(sns.heatmap(cf_matrix, center=True).set_title(" draw confusion matrix svc "))
plt.show()
joblib.dump(clf, "joblib_clf.pkl")
#logistic
print("----------------------------------------------------------- LOGISTIC--------------------------------------------------------- ")
ml = LogisticRegression(solver='liblinear', C=1.0, random_state=0)
ml.fit(X_train, y_train)
y_pred2 = ml.predict(X_test)
print("\n train score is :{:.3f} ".format(ml.score(X=X_train,y=y_train)))
print("test score is :{:.3f} ".format(ml.score(X=X_test,y=y_test)))
print(classification_report(y_test, y_pred2))
cm=confusion_matrix(y_test,y_pred2)
print("confusion_matrix :",cm)
sns.heatmap(cm,center=True).set_title("confusion matrices for all prediction features")
plt.show()
joblib.dump(ml, "joblib_ml.pkl")
#Decision Tree
print("---------------------------------------------------- DECISION TREE-------------------------------------------------------------- ")
cl = DecisionTreeClassifier( random_state=0)
cl.fit(X_train, y_train)
y_pred3 = cl.predict(X_test)
print("\n train score is : {:.3f}".format(cl.score(X=X_train, y=y_train)))
print("test score is : {:.3f}".format(cl.score(X=X_test, y=y_test)))
print(classification_report(y_test, y_pred3))
cm=confusion_matrix(y_test,y_pred3)
print("confusion_matrix :",cm)
sns.heatmap(cm,center=True).set_title("confusion matrices for all prediction features")
plt.show()
joblib.dump(cl, "joblib_cl.pkl")
from sklearn.metrics import log_loss
from sklearn.ensemble import VotingClassifier
model_1 = LogisticRegression()
model_2 = svm.LinearSVC()
model_3 = DecisionTreeClassifier()
final_model=VotingClassifier(
    estimators=[('lg', model_1), ('svm', model_2), ('id3', model_3)], voting='hard')
final_model.fit(X_train, y_train)
pre_final = final_model.predict(X_test)
print(log_loss(y_test, pre_final))
print(classification_report(y_test, pre_final))
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred)
print("confusion_matrix :",cm)
model_tree=DecisionTreeRegressor(max_depth=6,random_state=0)
model_tree.fit(X=X_train,y=y_train)
# importance feature
importance =model_tree.feature_importances_
print(importance)
def plot_feature_importance(model3):
    plt.figure(figsize=(8,6))
    n_feature=21
    plt.barh(range(n_feature),model3.feature_importances_,align='center')
    plt.yticks(np.arange(n_feature),x)
    plt.xlabel("feature importance")
    plt.xlabel("feature")
    plt.ylim(-1,n_feature)
plot_feature_importance(model_tree)
plt.savefig('feature')
plt.show()
dx=pd.read_csv("test.csv")
print(dx.shape)
dx = dx.dropna()
print(dx.shape)
dx=dx.drop_duplicates()
print(dx.shape)
x2=dx.iloc[:,1:22]
y2=dx['Diabetes_binary']
feature=SelectKBest(score_func=chi2,k=6)
#model=LinearRegression()
#feature=SelectFromModel(estimator=model)
x2=feature.fit_transform(x2,y2)
lg = joblib.load('/content/joblib_ml.pkl')
r1=lg.score(x2,y2)
z=lg.predict(x2)
print (classification_report(y2, z))
print(r1)
svm=joblib.load('/content/joblib_clf.pkl')
r2=svm.score(x2,y2)
z=svm.predict(x2)
print (classification_report(y2, z))
print(r2)
id3=joblib.load('/content/joblib_cl.pkl')
r3=id3.score(x2,y2)
z=id3.predict(x2)
print (classification_report(y2, z))
print(r3)