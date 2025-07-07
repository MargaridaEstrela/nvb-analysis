import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, GRU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# --- Config ---
DATA_DIR = "/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/pose_metrics_filtered/"
LABELS_FILE = "session_labels.json"
SEQUENCE_LENGTH = 400
STEP_SIZE = 200
NUM_CLASSES = 2

# These sessions have the excluded participant as _p1, so we need to swap _p0 with _p1
EXCLUDED_RIGHT = {"3", "5", "6", "17"}

# --- Load labels ---
with open(LABELS_FILE, "r") as f:
    session_labels = json.load(f)

# --- Combine features from both participants into paired samples ---
def load_combined_sessions(data_dir, labels_dict, window_size, step_size):
    X, y, session_ids = [], [], []
    
    swap_ids = {"3", "5", "6", "17"}

    for session_key in sorted(labels_dict.keys()):
        combined_path = os.path.join(data_dir, f"{session_key}")
        if not os.path.exists(combined_path):
            print(f"File not found: {combined_path}")
            continue

        label = labels_dict.get(session_key, None)
        if label is None:
            print(f"Label not found for session: {session_key}")
            continue

        df = pd.read_csv(combined_path).fillna(0.0)

        # Optional: drop confidence columns if still there
        df = df.loc[:, ~df.columns.str.endswith("_c")]

        # Split columns by participant
        p0_cols = [c for c in df.columns if c.endswith("_p0")]
        p1_cols = [c for c in df.columns if c.endswith("_p1")]

        df_p0 = df[p0_cols].copy()
        df_p1 = df[p1_cols].copy()

        # Rename columns
        df_p0.columns = [c.replace("_p0", "") for c in df_p0.columns]
        df_p1.columns = [c.replace("_p1", "") for c in df_p1.columns]
        
        # Detect if this session needs to swap participants
        session_id = session_key.replace("session", "").split("_")[0]
        if session_id in swap_ids:
            df_p0, df_p1 = df_p1, df_p0  # swap so excluded is always p0

        # Normalize per participant
        df_p0 = StandardScaler().fit_transform(df_p0)
        df_p1 = StandardScaler().fit_transform(df_p1)

        for start in range(0, len(df) - window_size + 1, step_size):
            end = start + window_size

            p0_window = df_p0[start:end]
            p1_window = df_p1[start:end]
            diff_window = p0_window - p1_window

            combined_window = np.concatenate([p0_window, p1_window, diff_window], axis=1)

            X.append(combined_window)
            y.append(label)
            session_id = session_key.split("_")[0]  # Extract session ID from key
            session_ids.append(session_id)

    return np.array(X), np.array(y), np.array(session_ids)


X, y, session_ids = load_combined_sessions(DATA_DIR, session_labels, SEQUENCE_LENGTH, STEP_SIZE)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)


# --- Train/Val split ---
best_split = None
best_diff = float("inf")

for _ in range(20):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=None)
    for train_idx, val_idx in splitter.split(X, y, groups=session_ids):
        y_val_tmp = y[val_idx]
        val_counts = np.bincount(y_val_tmp, minlength=NUM_CLASSES)
        if np.any(val_counts == 0):
            continue
        diff = np.std(val_counts)
        if diff < best_diff:
            best_diff = diff
            best_split = (train_idx, val_idx)

train_idx, val_idx = best_split
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y_cat[train_idx], y_cat[val_idx]


# --- Class weights ---
class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)


# --- Model ---
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, X.shape[2])),
    
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.2),
    
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),
    
    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5),
    ModelCheckpoint("best_model.keras", save_best_only=True)
]

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)


# --- Evaluation ---
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=[f"Class {i}" for i in range(NUM_CLASSES)], zero_division=0))


# --- Group-level evaluation ---
session_pred_probs = defaultdict(list)
session_true_labels = {}

for i, session_id in enumerate(session_ids[val_idx]):
    prob_class_1 = y_pred[i][1]
    session_pred_probs[session_id].append(prob_class_1)
    session_true_labels[session_id] = y_true[i]

aggregated_pred = []
aggregated_true = []
session_list = []

for session_id in sorted(session_pred_probs.keys(), key=lambda x: int(x.replace("session", ""))):
    mean_prob = np.mean(session_pred_probs[session_id])
    true_label = session_true_labels[session_id]

    aggregated_pred.append(mean_prob)
    aggregated_true.append(true_label)
    session_list.append(session_id)

pred_class = [int(p >= 0.5) for p in aggregated_pred]
print("\nSession-level Classification Report:")
print(classification_report(aggregated_true, pred_class))

# --- Plots ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='Train Acc')
ax[0].plot(history.history['val_accuracy'], label='Val Acc')
ax[0].set_title('Accuracy')
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].set_title('Loss')
ax[1].legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(aggregated_pred, label="Mean Predicted Probability (Class 1)", marker='o')
plt.plot(aggregated_true, label="True Label", linestyle='--', marker='x')
plt.xticks(ticks=np.arange(len(session_list)), labels=session_list, rotation=45)
plt.title("Session-level Prediction vs True Label")
plt.xlabel("Session")
plt.ylabel("Probability / Label")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()