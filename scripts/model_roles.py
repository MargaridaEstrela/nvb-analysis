import os
import json
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Bidirectional, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# --- Config ---
DATA_DIR = "/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/pose_metrics/"
LABELS_FILE = "role_labels.json"
SEQUENCE_LENGTH = 400
STEP_SIZE = 200
NUM_CLASSES = 3

# --- Load labels ---
with open(LABELS_FILE, "r") as f:
    session_labels = json.load(f)
    

# --- Fit scaler for each person ---
def fit_person_scaler(data_dir, labels_dict, exclude_cols=['frame']):
    all_features = []
    for file in labels_dict:
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        all_features.append(df[feature_cols].values)

    all_features = np.vstack(all_features)
    scaler = StandardScaler().fit(all_features)
    return scaler


# --- Sliding window creation ---
def create_windows(features, label, window_size, step_size):
    X, y = [], []
    for start in range(0, len(features) - window_size + 1, step_size):
        X.append(features[start:start+window_size])
        y.append(label)
    return X, y


# --- Load and preprocess data ---
def load_sessions(data_dir, labels_dict, scaler, window_size, step_size):
    X, y = [], []
    for file in labels_dict:
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)
        features = scaler.transform(df.values)
        windows, labels = create_windows(features, labels_dict[file], window_size, step_size)
        X.extend(windows)
        y.extend(labels)
    return np.array(X), np.array(y)


# --- Load and preprocess data by person ---
def load_sessions_by_person(data_dir, labels_dict, scaler, window_size, step_size):
    """
    Loads and preprocesses data for each person separately,
    applying scaling and creating sliding windows.
    """
    X, y = [], []

    for file, label in labels_dict.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)

        features = df.values

        # Apply scaling
        features = scaler.transform(features)

        # Create sliding windows
        windows, window_labels = create_windows(features, label, window_size, step_size)

        X.extend(windows)
        y.extend(window_labels)

    return np.array(X), np.array(y)

# Fit global scaler
scaler = fit_person_scaler(DATA_DIR, session_labels)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved to scaler.pkl")

# Load data with windows
X, y = load_sessions_by_person(DATA_DIR, session_labels, scaler, SEQUENCE_LENGTH, STEP_SIZE)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=y
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# --- Train-test split ---
# Dividing the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, stratify=y)


X_flat = X.reshape(X.shape[0], -1)
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
    X_flat, y, test_size=0.2, stratify=y, random_state=42
)
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train_rf, y_train_rf)
print("Random Forest accuracy:", clf.score(X_val_rf, y_val_rf))

importances = clf.feature_importances_

# Sort features
indices = np.argsort(importances)[::-1]
top_k = min(100, len(importances)) 


if hasattr(X, 'columns'):
    feature_names = X.columns
else:
    feature_names = [f"f{i}" for i in range(X_flat.shape[1])]


def generate_feature_names(seq_len, columns):
    return [col for _ in range(seq_len) for col in columns]

csv_columns_1 = pd.read_csv(os.path.join(DATA_DIR, "session31_1.csv"), nrows=1).columns.tolist()
full_feature_names = generate_feature_names(SEQUENCE_LENGTH, csv_columns_1)

aggregated = defaultdict(float)
for i, imp in enumerate(importances):
    name_base = full_feature_names[i]
    aggregated[name_base] += imp


# Select top K features
top_k = 30
agg_sorted = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_k]

agg_names = [name for name, _ in agg_sorted]
agg_values = [imp for _, imp in agg_sorted]

# --- Plot ---
plt.figure(figsize=(10, 8))
sns.barplot(x=agg_values, y=agg_names, palette='viridis')
plt.title("Top Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- Model ---
model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, X.shape[2])))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))

# Second Bi-LSTM layer
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))

# Third layer
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))

# Fully connected
model.add(Dense(64, activation='relu', kernel_regularizer=l2(3e-4)))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
)

# --- Callbacks ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

callbacks = [
    early_stop,
    ModelCheckpoint("best_model.keras", save_best_only=True)
]

# --- Training ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weight_dict
)

# --- Evaluate ---
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# --- Plot accuracy and loss ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax[0].plot(history.history['accuracy'], label='Train Acc')
ax[0].plot(history.history['val_accuracy'], label='Val Acc')
ax[0].set_title('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

# Loss plot
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].set_title('Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.show() 