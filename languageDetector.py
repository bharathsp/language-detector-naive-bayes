import pandas as pd
import re
import joblib  # recommended for sklearn model persistence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Load dataset
# ------------------------------
data = pd.read_csv("Language Detection.csv")
X = data["Text"]
y = data["Language"]

# Encode language labels to numeric values
le = LabelEncoder()
y = le.fit_transform(y)

# ------------------------------
# Text preprocessing function
# ------------------------------
def clean_text(text):
    """Remove unwanted characters and convert to lowercase."""
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)  # remove special chars & digits
    text = re.sub(r'[\[\]]', ' ', text)  # remove square brackets
    return text.lower()

X = X.apply(clean_text)

# ------------------------------
# Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------
# Define TF-IDF and Naive Bayes pipeline
# ------------------------------
tfidf = TfidfVectorizer(ngram_range=(1, 2))

# Naive Bayes with hyperparameter tuning
params = {'alpha': [0.01, 0.1, 0.5, 1]}
grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(tfidf.fit_transform(X_train), y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Final pipeline: vectorizer + classifier
pipe = Pipeline([
    ('vectorizer', tfidf),
    ('classifier', best_model)
])

# Train pipeline on full training data
pipe.fit(X_train, y_train)

# ------------------------------
# Save trained pipeline using joblib
# ------------------------------
joblib.dump(pipe, 'trained_pipeline-0.1.0.joblib')

# Save label encoder separately
joblib.dump(le, "label_encoder.joblib")

print("Model trained and saved as trained_pipeline-0.1.0.joblib")
