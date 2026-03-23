import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Preprocessing function
def preprocess(text):
    # lowercase
    text = text.lower()
    
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 2. Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", names=["label", "message"])

# 3. Apply preprocessing
data["clean_message"] = data["message"].apply(preprocess)

# 4. Features and labels
X = data["clean_message"]
y = data["label"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Vectorization
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 7. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Accuracy
print("Test Accuracy:", model.score(X_test, y_test))

# 9. Predict new messages
new_messages = [
    "hey",
    "Hey, are we meeting today?",
    "Claim your free money now!!!"
]

new_messages_clean = [preprocess(msg) for msg in new_messages]
new_X = vectorizer.transform(new_messages_clean)
predictions = model.predict(new_X)

# 10. Show results
for msg, pred in zip(new_messages, predictions):
    print("\nMessage:", msg)
    print("Cleaned:", preprocess(msg))
    print("Prediction:", pred)