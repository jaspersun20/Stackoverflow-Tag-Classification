import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercasing and removing special characters
    text = re.sub(r'\W', ' ', str(text).lower())
    # Tokenization
    tokens = text.split()
    # Stopword removal and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print("Loading dataset...")
df = pd.read_csv('full_dataset_v3.csv', delimiter=';')
print("Dataset loaded.")

print("Preprocessing text...")
df['title'] = df['title'].apply(preprocess_text)
print("Text preprocessing completed.")

# Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

# Split dataset (Note: we will split the data after vectorization for GridSearchCV)
X = df['title']

# Setting up the pipeline for Tfidf vectorization and Naive Bayes classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB())
])

# Hyperparameter tuning using GridSearchCV
parameters = {
    'nb__alpha': (0.001, 0.01, 0.1, 0.5, 0.8, 1, 2, 5, 10),
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print("Best Score: %s" % grid_search.best_score_)
print("Best Hyperparameters: %s" % grid_search.best_params_)

# Displaying the accuracy for each alpha tested
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%r => Mean accuracy: %0.5f (+/-%0.03f)" % (params, mean, std * 2))

# Evaluating the best model on the full dataset to get precision, recall, and F1 score
# Note: In a real-world scenario, you should split your data into a training and test set to evaluate these metrics.
y_pred = grid_search.predict(X)
report = classification_report(y, y_pred, target_names=encoder.classes_)
print(report)

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

# Use seaborn to visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()