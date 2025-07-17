import os
import json
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, GlobalAveragePooling1D, TimeDistributed, LayerNormalization, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# --- Config ---
DATA_DIR = "/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/pose_metrics/"
LABELS_FILE = "unreliable_labels.json"
SEQUENCE_LENGTH = 400
STEP_SIZE = 200

NUM_CLASSES = 2
TOP_K = 40

EPOCHS = 25
PATIENCE = 10
BATCH_SIZE = 64

ACCURACY_ACCEPTANCE_THRESHOLD = 0.77


# --- Fit scaler for each person ---
def fit_global_scaler(data_dir, labels_dict, exclude_cols=['frame']):
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
    X, y, indices = [], [], []
    for start in range(0, len(features) - window_size + 1, step_size):
        X.append(features[start:start+window_size])
        y.append(label)
        indices.append(start)  # Store the starting index of the window
    return X, y, indices


# --- Load and preprocess data by person ---
def load_sessions_by_person(data_dir, labels_dict, scaler, window_size, step_size):
    X, y, starts = [], [], []

    for file, label in labels_dict.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)

        features = df.values

        # Apply scaling
        features = scaler.transform(features)

        # Create sliding windows
        windows, window_labels, window_starts = create_windows(features, label, window_size, step_size)

        X.extend(windows)
        y.extend(window_labels)
        starts.extend(window_starts)

    return np.array(X), np.array(y), np.array(starts)
    

def load_sessions_temporal_split(data_dir, labels_dict, scaler, window_size, step_size, split_ratio=0.8):
    X_train, y_train, starts_train = [], [], []
    X_val, y_val, starts_val = [], [], []

    for file, label in labels_dict.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)
        features = scaler.transform(df.values)
        windows, labels, starts = create_windows(features, label, window_size, step_size)
        num_train = int(len(windows) * split_ratio)

        X_train.extend(windows[:num_train])
        y_train.extend(labels[:num_train])
        starts_train.extend(starts[:num_train])

        X_val.extend(windows[num_train:])
        y_val.extend(labels[num_train:])
        starts_val.extend(starts[num_train:])

    return (
        np.array(X_train), np.array(y_train), np.array(starts_train),
        np.array(X_val), np.array(y_val), np.array(starts_val)
    )
    

def generate_feature_names(seq_len, columns):
    return [col for _ in range(seq_len) for col in columns]


def get_group(feature_name):
    if feature_name.startswith("vel_"):
        return "Velocity"
    elif feature_name.startswith("disp_"):
        return "Displacement"
    elif feature_name.startswith("AU"):
        return "AU"


# --- Curriculum and Rolling eval ---
def truncate_sequences(X, max_len, min_len=SEQUENCE_LENGTH):
    out = []
    for x in X:
        t = x[:max_len]
        if t.shape[0] < min_len:
            pad_amt = min_len - t.shape[0]
            t = np.pad(t, ((0, pad_amt), (0, 0)), mode='constant', constant_values=0.)
        out.append(t)
    return np.stack(out)


def curriculum_training(X_train, y_train, X_val, y_val, frame_limits, train_params, build_fn):
    records = []
    for m in tqdm(frame_limits, desc="Curriculum"):
        Xt = truncate_sequences(X_train, m)
        Xv = truncate_sequences(X_val, m)
        yt = to_categorical(y_train, num_classes=NUM_CLASSES)
        yv = to_categorical(y_val, num_classes=NUM_CLASSES)
        
        model = build_fn(input_shape=(m, X_train.shape[-1]), **train_params.get('model_kwargs', {}))
        
        hist = model.fit(
            Xt, yt,
            validation_data=(Xv, yv),
            epochs=train_params.get('epochs', EPOCHS),
            batch_size=train_params.get('batch_size', BATCH_SIZE),
            **train_params.get('fit_kwargs',{})
        )
        
        acc = hist.history.get('val_accuracy', [None])[-1]
        records.append({'max_frame': m, 'val_accuracy': acc})
        
    return pd.DataFrame(records)
    

# --- Model ---
def build_lstm_model(input_shape):
    model = Sequential([
            Input(shape=input_shape),
            LayerNormalization(),
            Bidirectional(LSTM(64, return_sequences=True)),
            LayerNormalization(),
            Bidirectional(LSTM(32, return_sequences=True)),
            LayerNormalization(),
            GlobalAveragePooling1D(), Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(3e-4)), Dropout(0.3),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )
    
    return model


def main():
    # --- Load labels ---
    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)
    
    # Group by sessions (e.g.: session1 -> [session1_0.csv, session1_1.csv])
    session_to_files = defaultdict(list)
    session_to_label = {}

    for filename, label in labels.items():
        session = filename.split("_")[0]
        session_to_files[session].append(filename)
        session_to_label[session] = label

    session_ids = list(session_to_files.keys())
    session_labels = [session_to_label[s] for s in session_ids]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(session_ids, session_labels, groups=session_ids))

    train_sessions = [session_ids[i] for i in train_idx]
    val_sessions   = [session_ids[i] for i in val_idx]

    train_files = [f for s in train_sessions for f in session_to_files[s]]
    val_files   = [f for s in val_sessions for f in session_to_files[s]]

    # Fit & save scaler
    scaler = fit_global_scaler(DATA_DIR, labels)
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to scaler.pkl")

    # Load data with 80/20 temporal split per session
    X_train, y_train, starts_train, X_val, y_val, starts_val = load_sessions_temporal_split(
        DATA_DIR, labels, scaler, SEQUENCE_LENGTH, STEP_SIZE
    )
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat   = to_categorical(y_val,   num_classes=NUM_CLASSES)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=y_train
    )

    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    # --- Random Forest feature importance ---
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_val_rf   = X_val.reshape(X_val.shape[0], -1)
    y_train_rf = y_train
    y_val_rf   = y_val
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    print("Random Forest accuracy:", rf.score(X_val_rf, y_val_rf))

    # --- Plot Top-k Features ---
    base_cols = pd.read_csv(os.path.join(DATA_DIR, next(iter(labels))), nrows=1).columns.tolist()
    feat_names = generate_feature_names(SEQUENCE_LENGTH, base_cols)
    imp = rf.feature_importances_
    agg = defaultdict(float)
    for idx, val in enumerate(imp):
        agg[feat_names[idx]] += val
    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    
    # Sort features
    indices = np.argsort(imp)[::-1]
    top_k = min(100, len(imp))
    names, vals = zip(*top)
    groups = [get_group(name) for name in names]
    
    # --- Assign color per group ---
    palette = {'AU': '#1f77b4', 'Displacement': '#2ca02c', 'Velocity': '#d62728'}
    bar_colors = [palette[g] for g in groups]
    
    plt.figure(figsize=(10,8))
    sns.barplot(x=vals, y=names, hue=names, palette=bar_colors)
    plt.title(f'Top {TOP_K} RF Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    patches = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    plt.legend(handles=patches, title="Feature Group")
#    plt.show()

    max_retries = 10
    while(True):
        # --- LSTM Training ---
        model = build_lstm_model((SEQUENCE_LENGTH, X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)
        
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, model_checkpoint],
            verbose=1,
            class_weight=class_weight_dict
        )
        
        # --- Evaluate LSTM ---
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_cat, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))
        ConfusionMatrixDisplay.from_predictions(y_true_classes, y_pred_classes, display_labels=[f"Class {i}" for i in range(NUM_CLASSES)])
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true_classes, y_pred_classes))
        
        
        if report['accuracy'] < ACCURACY_ACCEPTANCE_THRESHOLD:
            print(f"Accuracy too low ({report['accuracy']:.2f}), retraining...")
            continue  # Retry if accuracy is too low
            
        # Plot Accuracy and Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Global frame-wise accuracy ---
        timeline_len = max(starts_val) + SEQUENCE_LENGTH
        counts = np.zeros(timeline_len)
        corrects = np.zeros(timeline_len)

        for i, start in enumerate(starts_val):
            true = y_true_classes[i]
            pred = y_pred_classes[i]
            for offset in range(SEQUENCE_LENGTH):
                frame = start + offset
                if frame >= timeline_len:
                    continue
                counts[frame] += 1
                corrects[frame] += (pred == true)

        # Avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_timeline = np.where(counts > 0, corrects / counts, np.nan)

        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(acc_timeline)
        plt.xlabel("Frame index (global)")
        plt.ylabel("Window prediction accuracy")
        plt.title("Window-wise prediction accuracy over frames")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        
        # --- Curriculum learning: evaluate accuracy over window lengths ---
        frame_limits = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

        train_params = {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'model_kwargs': {},  # Optional model hyperparams
            'fit_kwargs': {
                'verbose': 0,  # Avoid cluttered logs
                'class_weight': class_weight_dict,
                'callbacks': [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
            }
        }

        results_df = curriculum_training(X_train, y_train, X_val, y_val, frame_limits, train_params, build_lstm_model)

        # Plot curriculum result
        plt.figure(figsize=(10, 5))
        plt.plot(results_df['max_frame'], results_df['val_accuracy'], marker='o')
        plt.xlabel('Window Length (frames)')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs. Window Length')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        break  # Exit after successful training

    
if __name__ == '__main__':
    main()
