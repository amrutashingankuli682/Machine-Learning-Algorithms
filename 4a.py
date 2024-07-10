#Demonstrate the working of SVM Classifier for suitable dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y =data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Support Vector Classifier
svm_classifier = SVC(kernel='linear').fit(X_train, y_train)

# Make predictions with SVC
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Evaluate the model
print('Accuracy:', accuracy)

cr=classification_report(y_test, y_pred)
print('Classification Report:\n ',cr)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n",cm)
