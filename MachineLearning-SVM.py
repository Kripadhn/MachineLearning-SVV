from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create the SVM model
svm = SVC(kernel='linear', C=1)

# Fit the model to the training data
svm.fit(X_train, y_train)

# Test the model on the test data
score = svm.score(X_test, y_test)
print(score)
