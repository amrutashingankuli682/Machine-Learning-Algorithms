# Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_score
# Setting up the dataset
data = [
    ("I love this movie", "positive"),
    ("This film was amazing", "positive"),
    ("I actually enjoyed it", "positive"),
    ("I hate this movie", "negative"),
    ("This film was terrible", "negative"),
    ("I did not like it", "negative")
]

texts, labels = zip(*data)

# Initializing the Models
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

classifier = MultinomialNB()
classifier.fit(X, labels)

# Testing the model and displaying the results
test_texts = [
    "I love this film",
    "I hated the movie",
    "It was an awesome movie",
    "This movie was not good"
]

X_test = vectorizer.transform(test_texts)
predictions = classifier.predict(X_test)

for text, label in zip(test_texts, predictions):
    print(f"Text: '{text}' => Predicted Label: '{label}'")
accuracy = accuracy_score(labels, classifier.predict(X))
print(f"\nAccuracy on training data: {accuracy:.2f}")
 