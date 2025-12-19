import pandas as pd
import numpy as np
import cv2
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===============================
# 1. LOAD DATASET
# ===============================
file_path = r"C:\Users\Alen\Python Projects\Mini Python Projects\DataSets\spam.csv"

df = pd.read_csv(file_path, encoding="latin-1")

print("Original Columns:", df.columns.tolist())
print(df.head(), "\n")


# ===============================
# 2. CLEAN DATASET
# ===============================
# Drop index column
df = df.iloc[:, 1:]   # keep spamORham, Message
df.columns = ["label", "message"]

# Encode labels
df["label"] = df["label"].str.lower().map({
    "ham": 0,
    "spam": 1
})

# Drop invalid rows
df.dropna(inplace=True)

print("Label Distribution:")
print(df["label"].value_counts(), "\n")


# ===============================
# 3. TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df["clean_message"] = df["message"].apply(clean_text)


# ===============================
# 4. TRAIN-TEST SPLIT
# ===============================
X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ===============================
# 5. TF-IDF VECTORIZATION
# ===============================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ===============================
# 6. TRAIN MODEL
# ===============================
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# ===============================
# 7. EVALUATION
# ===============================
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ===============================
# 8. CONFUSION MATRIX (OpenCV)
# ===============================
cm = confusion_matrix(y_test, y_pred)

img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (40, 40), (260, 260), (255, 255, 255), 2)

labels = [("TN", cm[0][0]), ("FP", cm[0][1]),
          ("FN", cm[1][0]), ("TP", cm[1][1])]

positions = [(60, 120), (160, 120), (60, 200), (160, 200)]

for (label, value), pos in zip(labels, positions):
    cv2.putText(
        img,
        f"{label}: {value}",
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

cv2.imshow("Spam Email Detection - Confusion Matrix", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ===============================
# 9. USER INPUT PREDICTION
# ===============================
def predict_email(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    spam_prob = prob[1]

    if spam_prob >= 0.45:
        return f"SPAM  ({spam_prob:.2f})"
    else:
        return f"HAM   ({1 - spam_prob:.2f})"


print("\n🔍 Spam Email Checker")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Enter a message to check: ")

    if user_input.lower() == "exit":
        print("Exiting Spam Checker 👋")
        break

    print("Result:", predict_email(user_input), "\n")
