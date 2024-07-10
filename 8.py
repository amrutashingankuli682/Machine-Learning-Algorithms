import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
df = pd.read_csv("diabetes.csv")
feature_col_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age'] 
predicted_class_names = ['Outcome']
X = df[feature_col_names].values # these are factors for the prediction 
y = df[predicted_class_names].values # this is what we want to predict #splitting the dataset into train and test data 
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)
print ('\n The total number of Training Data:',ytrain.shape) 
print ('\n The total number of Test Data:',ytest.shape)
# Training Naive Bayes (NB) classifier on training data. 
clf = GaussianNB().fit(xtrain,ytrain.ravel())
predicted = clf.predict(xtest)
predictTestData= clf.predict([[6,148,72,35,0,33.6,50]]) #printing Confusion matrix, accuracy, Precision and Recall 
print('\n Confusion matrix') 
print(metrics.confusion_matrix(ytest,predicted))
# Calculate and print accuracy
accuracy = metrics.accuracy_score(ytest, predicted)
print('\n Accuracy of the classifier:', accuracy)

# Calculate and print precision
precision = metrics.precision_score(ytest, predicted)
print('\n The value of Precision:', precision)

# Calculate and print recall
recall = metrics.recall_score(ytest, predicted)
print('\n The value of Recall:', recall)

print("Predicted Value for individual Test Data:", predictTestData)
