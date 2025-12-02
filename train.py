import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA = 'dataset_10k.csv'
OUT_MODEL = 'models/neformal_svm_pipeline.joblib'

print('Loading dataset...')
df = pd.read_csv(DATA)
X = df['text'].astype(str)
y = df['label'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
    ('clf', CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=5))
])

print('Training SVM (this may take a moment)...')
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)
print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, OUT_MODEL)
print('Saved model to', OUT_MODEL)
