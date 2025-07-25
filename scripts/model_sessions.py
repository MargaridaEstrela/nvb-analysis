import os
import json
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, GlobalAveragePooling1D, TimeDistributed, LayerNormalization, GlobalMaxPooling1D, BatchNormalization, Conv1D, Activation, Multiply, Permute, Lambda, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression

# --- Config ---
DATA_DIR = "/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/pose_metrics/"
LABELS_FILE = "session_labels.json"
SEQUENCE_LENGTH = 400
STEP_SIZE = 200

NUM_CLASSES = 2
TOP_K = 60

N_SPLITS = 5
EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 64

REPEATS = 10
TEST_SIZE = 0.2
SEED = 42
TEST_THRESHOLD = 0.8

def _prepare_features(df, feature_list):
    if feature_list is None:
        return df
    for f in feature_list:
        if f not in df.columns:
            df[f] = 0.0
    return df[feature_list]

# --- Fit scaler for each person ---
def fit_global_scaler(data_dir, labels_dict, exclude_cols=['frame'], selected_features=None):
    all_features = []
    for file in labels_dict:
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)

        if selected_features:
            for f in selected_features:
                if f not in df.columns:
                    df[f] = 0.0
            df = df[selected_features]
            
        all_features.append(df.values)

    all_features = np.vstack(all_features)
    scaler = StandardScaler().fit(all_features)
    return scaler


# --- Sliding window creation ---
def create_windows(features, label, window_size, step_size):
    X, y, indices = [], [], []
    for start in range(0, len(features) - window_size + 1, step_size):
        X.append(features[start:start+window_size])
        y.append(label)
        indices.append(start)
    return X, y, indices


# --- Load and preprocess data by person ---
def load_sessions_by_person(data_dir, labels_dict, scaler, window_size, step_size, top_feature_names=None):
    X, y, starts, session_ids = [], [], [], []

    for file, label in labels_dict.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)

        # Filter before scaling
        df = _prepare_features(df, top_feature_names)

        # Scale
        if scaler is not None:
            if df.shape[1] != scaler.n_features_in_:
                raise ValueError(f"Scaler expects {scaler.n_features_in_} features but got {df.shape[1]} in {file}")
            df_scaled = scaler.transform(df.values)
            df = pd.DataFrame(df_scaled, columns=df.columns)

        features = df.values

        # Create sliding windows
        windows, window_labels, window_starts = create_windows(features, label, window_size, step_size)

        X.extend(windows)
        y.extend(window_labels)
        starts.extend(window_starts)

        session_name = file.split("_")[0]
        session_ids.extend([session_name] * len(windows))

    return np.array(X), np.array(y), np.array(starts), np.array(session_ids)
    

def load_sessions_temporal_split(data_dir, labels_dict, scaler, window_size, step_size, split_ratio=0.8, top_feature_names=None):
    X_train, y_train, starts_train = [], [], []
    X_val, y_val, starts_val = [], [], []

    for file, label in labels_dict.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)

        # Filter before scaling
        df = _prepare_features(df, top_feature_names)

        # Scale
        if scaler is not None:
            if df.shape[1] != scaler.n_features_in_:
                raise ValueError(f"Scaler expects {scaler.n_features_in_} features but got {df.shape[1]} in {file}")
            df_scaled = scaler.transform(df.values)
            df = pd.DataFrame(df_scaled, columns=df.columns)

        features = df.values

        # Create windows
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
        
# --- Feature selection using Random Forest ---
def select_top_k_features(X, y, k):
    # Flatten time dimension for RF: (samples, time * features)
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, n_timesteps * n_features))
    
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_flat, y)
    
    importances = rf.feature_importances_
    top_k_indices = np.argsort(importances)[-k:]

    return top_k_indices, importances
    
    
# --- Select top-k features from the dataset ---
def select_features(X, top_k_indices):
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, n_timesteps * n_features))
    X_topk_flat = X_flat[:, top_k_indices]
    return X_topk_flat.reshape((n_samples, n_timesteps, -1))
    

def plot_logistic_regression_scree(importance_df):
    importance_df = importance_df.copy()
    importance_df['abs_weight'] = importance_df['weight'].abs()
    importance_df = importance_df.sort_values(by='abs_weight', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 5))
    plt.plot(importance_df['abs_weight'].values, marker='o')
    plt.title("Scree plot of feature importances (Logistic Regression)")
    plt.xlabel("Feature rank")
    plt.ylabel("Absolute coefficient value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

# --- Run logistic regression to find top features ---
def run_session_logistic_regression(labels, pose_metrics_dir):
    X_rows, y_rows = [], []
    valid_files = []

    for file, label in labels.items():
        try:
            df = pd.read_csv(os.path.join(pose_metrics_dir, file)).fillna(0.0).drop(columns=["frame"], errors="ignore")

            stats = pd.concat([df.mean(), df.std()], axis=0)
            X_rows.append(stats.values)
            y_rows.append(label)
            valid_files.append(file)
        except:
            continue

    X = np.vstack(X_rows)
    y = np.array(y_rows)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"Logistic Regression — Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # Top features
    feature_names = list(df.columns) + [f"{col}_std" for col in df.columns]
    weights = model.coef_[0]
    importance = pd.DataFrame({'feature': feature_names, 'weight': weights}).sort_values(by='weight', key=abs, ascending=False)
    
    top_features = importance["feature"].iloc[:TOP_K].tolist()
    
    print(f"Top {TOP_K} features:")
    print(importance.head(TOP_K))
    
    def strip_feature_names(features):
        base_features = set()
        for f in features:
            if f.endswith('_mean') or f.endswith('_std'):
                base_features.add(f.rsplit('_', 1)[0])
            else:
                base_features.add(f)
        return sorted(list(base_features))  # sort for consistency
        
    filtered_feature_names = strip_feature_names(top_features)
    print(f"\nBase features for time-series filtering: {filtered_feature_names}")
    
    # plot_logistic_regression_scree(importance)
    
    return filtered_feature_names
    

# --- Model ---
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LayerNormalization(),
        # Bidirectional(LSTM(128, return_sequences=True)),
        # LayerNormalization(),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        GlobalAveragePooling1D(),
        # Dense(64, activation='relu'),
        # Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )

    model.summary()
    return model


def extract_static_features_for_clustering(labels, data_dir, top_feature_names=None, scaler=None):
    X = []
    y = []
    files = []

    for file, label in labels.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)
        
        if top_feature_names:
            for f in top_feature_names:
                if f not in df.columns:
                    df[f] = 0.0
            df = df[top_feature_names]

        if scaler:
            df = pd.DataFrame(scaler.transform(df), columns=df.columns)

        # Aggregate statistics
        stats = pd.concat([df.mean(), df.std()], axis=0)  # (features * 2,)
        X.append(stats.values)
        y.append(label)
        files.append(file)

    return np.vstack(X), np.array(y), files



def cluster_and_plot(X, y, method='kmeans', n_clusters=2):
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = clusterer.fit_predict(X)

    # Evaluate clustering
    ari = adjusted_rand_score(y, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.4f} (0 = random, 1 = perfect)")

    # Reduce dimensionality for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 5))

    # Color by cluster
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred, palette="tab10")
    plt.title("Clustering (unsupervised)")

    # Color by true label
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1")
    plt.title("Ground truth: Fault-Exposed (1) vs Fault-Free (0)")

    plt.tight_layout()
    plt.show()
    


def main():
    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)
        
    top_feature_names = run_session_logistic_regression(labels, DATA_DIR)
        
    # --- Group files by session ---
    session_to_files = defaultdict(list)
    session_to_label = {}

    for filename, label in labels.items():
        session = filename.split("_")[0]
        session_to_files[session].append(filename)
        session_to_label[session] = label

    session_ids = list(session_to_files.keys())
    session_labels = [session_to_label[s] for s in session_ids]
    
    all_accuracies = []
    all_train_losses = []
    all_val_losses = []
    all_f1s = []

    for repeat in range(REPEATS):
        print(f"\n🔁 Repetition {repeat + 1}/{REPEATS}")

        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        
        for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(session_ids, session_labels, groups=session_ids)):
            print(f"— Fold {fold + 1}/{N_SPLITS}")
            
            # --- Split sessions into train/val and test sets ---
            train_val_sessions = [session_ids[i] for i in train_val_idx]
            train_val_files = [f for s in train_val_sessions for f in session_to_files[s]]
            train_val_labels = {f: labels[f] for f in train_val_files}
            
            test_sessions = [session_ids[i] for i in test_idx]
            test_files = [f for s in test_sessions for f in session_to_files[s]]
            test_labels = {f: labels[f] for f in test_files}

            # --- Fit scaler ---
            scaler = fit_global_scaler(DATA_DIR, train_val_labels, selected_features=top_feature_names)
            joblib.dump(scaler, f"scaler_repeat_{repeat+1}.pkl")
            
            X, y, files = extract_static_features_for_clustering(
                labels,
                data_dir=DATA_DIR,
                top_feature_names=top_feature_names,
                scaler=scaler
            )
            # cluster_and_plot(X, y)

            # --- Temporal split ---
            X_train, y_train, _, X_val, y_val, _ = load_sessions_temporal_split(
                DATA_DIR, train_val_labels, scaler, SEQUENCE_LENGTH, STEP_SIZE, split_ratio=0.8, top_feature_names=top_feature_names
            )

            # --- Test set ---
            X_test, y_test, starts, test_session_ids = load_sessions_by_person(
                DATA_DIR, test_labels, scaler, SEQUENCE_LENGTH, STEP_SIZE, top_feature_names=top_feature_names
            )

            # --- Class weights ---
            class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))

            # --- Build & train model ---
            model = build_lstm_model((SEQUENCE_LENGTH, X_train.shape[2]))
            early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
            checkpoint = ModelCheckpoint(f"best_model.keras", save_best_only=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stop, checkpoint, reduce_lr],
                verbose=1,
                class_weight=class_weight_dict
            )

            # --- Evaluate ---
            y_pred = model.predict(X_test)
            y_pred_classes = (y_pred >= 0.5).astype(int).flatten()
            
            # --- Store time-slot accuracy ---
            window_results = pd.DataFrame({
                "session": test_session_ids,
                "start": starts,
                "true": y_test,
                "pred": y_pred_classes,
                "score": y_pred.flatten()
            })
            window_results["correct"] = (window_results["true"] == window_results["pred"]).astype(int)

            # --- Find best time slots ---
            time_slot_accuracy = window_results.groupby(["session", "start"])["correct"].mean().reset_index()
            top_time_slots = time_slot_accuracy.sort_values(by="correct", ascending=False).head(10)

            print("\n🔍 Best time slots across sessions:")
            print(top_time_slots)
            
            # plt.figure(figsize=(12, 6))
            # sns.lineplot(data=time_slot_accuracy, x="start", y="correct", hue="session", marker="o")
            # plt.title("Window Accuracy Over Time (per session)")
            # plt.xlabel("Window Start Frame")
            # plt.ylabel("Accuracy (0 or 1)")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
            
            acc = accuracy_score(y_test, y_pred_classes)
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            all_accuracies.append(acc)
            all_train_losses.append(history.history['loss'][-1])
            all_val_losses.append(history.history['val_loss'][-1])
            all_f1s.append(f1)

            # print(f"Repeat {repeat+1}, Fold {fold+1} — Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
            print(classification_report(y_test, y_pred_classes))
            print(confusion_matrix(y_test, y_pred_classes))
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_classes), display_labels=np.unique(y_test))
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - Repeat {repeat+1}")
            # plt.title(f"Confusion Matrix - Repeat {repeat+1}, Fold {fold+1}")
            
            # # --- Training curves ---
            # plt.figure(figsize=(12, 6))
            
            # plt.subplot(1, 2, 1)
            # plt.plot(history.history['accuracy'], label='Train Accuracy')
            # plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            # plt.title('Accuracy Curves')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # plt.legend()
            
            # plt.subplot(1, 2, 2)
            # plt.plot(history.history['loss'], label='Train Loss')
            # plt.plot(history.history['val_loss'], label='Val Loss')
            # plt.title('Loss Curves')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
    
            # plt.tight_layout()
            # plt.show()

                
        # --- Results Folds
        

    # --- Final results ---
    print("\nFinal results across repetitions:")
    print(f"Mean Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Mean Train Loss: {np.mean(all_train_losses):.4f} ± {np.std(all_train_losses):.4f}")
    print(f"Mean Val Loss: {np.mean(all_val_losses):.4f} ± {np.std(all_val_losses):.4f}")
    print(f"Mean F1 Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    
if __name__ == '__main__':
    main()
