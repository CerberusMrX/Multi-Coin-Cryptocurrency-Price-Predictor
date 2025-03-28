from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv('featured_data.csv')
features = ['Open', 'Close', 'Volume', 'Coin_Encoded', 'SMA10', 'RSI', 'MACD', 'BB_Mid', 'BB_Upper', 'BB_Lower']
X = df[features]
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)
y = df['Target']

train_size = int(len(X_scaled) * 0.8)
X_test = X_scaled[train_size:]
y_test = y[train_size:]

model = joblib.load('crypto_model.pkl')
y_pred = model.predict(X_test)

# Metrics
logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Feature Importance
importances = model.feature_importances_
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
