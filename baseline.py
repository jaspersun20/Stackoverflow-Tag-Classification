import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

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

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('full_dataset_v3.csv', delimiter=';')
print("Dataset loaded.")

# Preprocess titles
print("Preprocessing text...")
df['title'] = df['title'].apply(preprocess_text)
print("Text preprocessing completed.")

# Feature Extraction
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for efficiency
X = vectorizer.fit_transform(df['title']).toarray()
print("Feature extraction completed.")

# Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training completed.")

# Model Evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=encoder.classes_))

# For large datasets, tqdm can be integrated to visualize progress. Here's an example applied to text preprocessing:
df['title'] = [preprocess_text(title) for title in tqdm(df['title'])]
