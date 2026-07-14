import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ---- Step 1: Dataset load karo ----
url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv"
df = pd.read_csv(url)

# ---- Step 2: Features (X) aur Label (y) alag karo ----
X = df.drop('phishing', axis=1)   # sab columns except 'phishing'
y = df['phishing']                # sirf label column (0 = legit, 1 = phishing)

# ---- Step 3: Train/Test split (80% train, 20% test) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Step 4: Model banao aur train karo ----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Step 5: Predictions aur evaluation ----
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---- Step 6: Model save karo (future use ke liye) ----
joblib.dump(model, 'phishing_model.pkl')
print("\nModel saved as phishing_model.pkl")

# ---- Step 7: Confusion Matrix ko graph bana ke save karo ----
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Phishing'])
disp.plot(cmap='Blues')
plt.title('Phishing Detector - Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix graph saved as confusion_matrix.png")