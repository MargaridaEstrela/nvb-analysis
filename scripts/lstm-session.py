import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# --- Config ---
DATA_DIR = "/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/pose_metrics_truncated/relative/"
LABELS_FILE = "session_labels.json"
FEATURES_JSON = "features_session-level.json"
FOLDER_NAME = "session-level-lstm"

REAL_TIME_MODE = True

# LSTM optimized config
SEQUENCE_LENGTH = 300
STEP_SIZE = 150

N_SPLITS = 3
REPEATS = 5
SEED = 42

EPOCHS = 20
BATCH_SIZE = 8
PATIENCE = 3

EXPERIMENT_GROUP = "full"  # "full", "aus", "motion", "interpersonal"

def load_selected_features_from_json(filename=FEATURES_JSON):
    """Load selected features from your JSON file and remove duplicates"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            features = json.load(f)
        
        # Remove duplicates while preserving order
        unique_features = list(dict.fromkeys(features))
        
        if len(unique_features) != len(features):
            duplicates = [x for i, x in enumerate(features) if x in features[:i]]
            print(f"Removed {len(features) - len(unique_features)} duplicate features: {duplicates}")
        
        print(f"Loaded {len(unique_features)} unique features from {filename} (originally {len(features)})")
        return unique_features
    else:
        raise FileNotFoundError(f"Feature file {filename} not found!")

def select_features_from_json(available_features, json_filename=FEATURES_JSON):
    """Select features from JSON file, mapping to pair_mean__ and pair_abs__ versions"""
    target_features = load_selected_features_from_json(json_filename)
    
    selected_features = []
    missing_features = []
    
    print(f"Mapping {len(target_features)} features from JSON...")
    
    for i, target_feat in enumerate(target_features):
        # Try different variations to match your dataset format
        variations = [
            target_feat,                    # Exact match
            f"pair_mean__{target_feat}",   # Most likely format
            f"pair_abs__{target_feat}",    # Alternative format
        ]
        
        found = False
        for variation in variations:
            if variation in available_features and variation not in selected_features:
                selected_features.append(variation)
                found = True
                break
        
        if not found:
            missing_features.append(target_feat)
    
    print(f"Found {len(selected_features)}/{len(target_features)} unique features from JSON in dataset")
    
    if len(selected_features) != len(set(selected_features)):
        print("WARNING: Duplicate features detected!")
        duplicates = [x for i, x in enumerate(selected_features) if x in selected_features[:i]]
        print(f"Duplicate features: {duplicates}")
        # Remove duplicates while preserving order
        selected_features = list(dict.fromkeys(selected_features))
        print(f"After removing duplicates: {len(selected_features)} features")
    
    if missing_features:
        print(f"Missing features ({len(missing_features)}): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    
    if len(selected_features) == 0:
        raise ValueError("No features from JSON file found in dataset! Check feature names.")
    
    print(f"Final: Using {len(selected_features)} unique features from {json_filename}")
    
    return selected_features

def select_feature_group(df, group):
    """Filter features by group type"""
    df = df.drop(columns=["frame", "window_id"], errors="ignore")
    df = df.select_dtypes(include=[np.number]).fillna(0.0)

    def inner_token(name: str) -> str:
        if name.startswith("pair_mean__") or name.startswith("pair_abs__"):
            return name.split("__", 1)[1]
        return name

    motion_prefixes = (
        "disp_", "vel_", "xdrift_", "ydrift_", "zdrift_",
        "shoulder_yaw", "head_v", "hand_raise_", "reach_"
    )

    if group == "aus":
        keep = [c for c in df.columns if inner_token(c).lower().startswith("au")]
    elif group == "motion":
        keep = [c for c in df.columns if inner_token(c).lower().startswith(motion_prefixes)]
    elif group == "interpersonal":
        keep = [c for c in df.columns if inner_token(c).lower().startswith(("d_centroid", "delta_"))]
    else:  # "full"
        keep = list(df.columns)

    return df.reindex(columns=sorted(keep), fill_value=0.0)

def get_feature_names_for_split(data_dir, files, group):
    """Get all feature names for a given group across files"""
    feats = set()
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f)).fillna(0.0)
        df = select_feature_group(df, group)
        feats.update(df.columns.tolist())
    return sorted(list(feats))

def fit_global_scaler(data_dir, labels_dict, selected_features=None):
    """Fit scaler across all sessions"""
    all_features = []
    for file in labels_dict:
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)
        df = select_feature_group(df, EXPERIMENT_GROUP)
        
        if selected_features:
            # Ensure we only use the selected features
            df = _prepare_features(df, selected_features)
            
        all_features.append(df.values)

    all_features = np.vstack(all_features)
    scaler = StandardScaler().fit(all_features)
    return scaler

def create_windows(features, label, window_size, step_size, real_time_mode=False):
    """Create sliding windows from session data - adaptable for real-time detection"""
    X, y, indices = [], [], []
    
    if real_time_mode:
        # For real-time: create overlapping windows throughout the session
        for start in range(0, len(features) - window_size + 1, step_size):
            X.append(features[start:start+window_size])
            y.append(label)  # Same label for all windows in session
            indices.append(start)
    else:
        # Current behavior: session-level windows
        for start in range(0, len(features) - window_size + 1, step_size):
            X.append(features[start:start+window_size])
            y.append(label)
            indices.append(start)
    
    return X, y, indices

def _prepare_features(df, feature_list):
    """Prepare features by selecting only specified features"""
    df = df.drop(columns=["frame", "window_id"], errors="ignore")
    
    # Ensure we only get the exact features requested
    missing_features = []
    for f in feature_list:
        if f not in df.columns:
            df[f] = 0.0  # Add missing features as zeros
            missing_features.append(f)
    
    if missing_features:
        print(f"    Added {len(missing_features)} missing features as zeros")
    
    # Select ONLY the requested features in the exact order
    df_selected = df[feature_list]
    
    return df_selected

def load_sessions(data_dir, labels_dict, scaler, window_size, step_size, top_feature_names=None):
    """Load session data and create windows"""
    X, y, starts, session_ids = [], [], [], []

    for file, label in labels_dict.items():
        df = pd.read_csv(os.path.join(data_dir, file)).fillna(0.0)
        
        df = select_feature_group(df, EXPERIMENT_GROUP)
        df = _prepare_features(df, top_feature_names)

        # Scale
        if scaler is not None:
            if df.shape[1] != scaler.n_features_in_:
                raise ValueError(f"Scaler expects {scaler.n_features_in_} features but got {df.shape[1]} in {file}")
            df_scaled = scaler.transform(df.values)
            df = pd.DataFrame(df_scaled, columns=df.columns)

        features = df.values
        windows, window_labels, window_starts = create_windows(
            features, label, window_size, step_size, real_time_mode=REAL_TIME_MODE
        )

        X.extend(windows)
        y.extend(window_labels)
        starts.extend(window_starts)

        session_name = file.split("_")[0]
        session_ids.extend([session_name] * len(windows))

    return np.array(X), np.array(y), np.array(starts), np.array(session_ids)

def build_lstm_model(sequence_length, n_features):
    """Build LSTM model for sequence classification"""
    model = Sequential([
        Input(shape=(sequence_length, n_features)),
        LSTM(32, return_sequences=True, dropout=0.5),
        BatchNormalization(),
        LSTM(16, return_sequences=False, dropout=0.5),
        BatchNormalization(),
        Dense(8, activation='relu', kernel_regularizer=l2(0.02)),
        Dropout(0.7),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_lstm_model(X_train, X_test, y_train, y_test):
    """Train LSTM model with loss tracking"""
    print(f"Training LSTM with {len(X_train)} samples, testing with {len(X_test)} samples")
    print(f"Input shape: {X_train.shape}")
    
    # Build model
    model = build_lstm_model(X_train.shape[1], X_train.shape[2])
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='loss', 
        patience=PATIENCE, 
        restore_best_weights=True,
        min_delta=0.01,
        verbose=1
    )
    
    # Train model
    print("Training LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, len(X_train)),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Extract loss information
    final_train_loss = history.history['loss'][-1]
    epochs_trained = len(history.history['loss'])
    
    print(f"Training completed after {epochs_trained} epochs")
    print(f"Final train loss: {final_train_loss:.4f}")
    
    return model, history
def extract_temporal_metrics(results_df, fps=30):
    """Extract timing-based metrics for real-time detection"""
    temporal_metrics = {}
    
    # Detection delay analysis
    exclusion_sessions = results_df[results_df['true_label'] == 1]['session'].unique()
    detection_delays = []
    detection_rates = []
    
    for session in exclusion_sessions:
        session_data = results_df[results_df['session'] == session].sort_values('start_frame')
        first_positive_window = session_data[session_data['pred_label'] == 1]['start_frame'].min()
        
        if not pd.isna(first_positive_window):
            # Calculate delay in seconds
            delay_seconds = first_positive_window / fps
            detection_delays.append(delay_seconds)
            
            # Detection rate for this session
            pred_positive = np.sum(session_data['pred_label'])
            total_windows = len(session_data)
            detection_rates.append(pred_positive / total_windows)
    
    temporal_metrics['mean_detection_delay_seconds'] = np.mean(detection_delays) if detection_delays else np.nan
    temporal_metrics['median_detection_delay_seconds'] = np.median(detection_delays) if detection_delays else np.nan
    temporal_metrics['detection_success_rate'] = len(detection_delays) / len(exclusion_sessions) if exclusion_sessions.size > 0 else 0
    temporal_metrics['mean_session_detection_rate'] = np.mean(detection_rates) if detection_rates else np.nan
    
    return temporal_metrics

def extract_confidence_metrics(y_test, y_pred_proba):
    """Analyze prediction confidence patterns"""
    confidence_metrics = {}
    
    # Confidence distribution by class
    positive_class_conf = y_pred_proba[y_test == 1]
    negative_class_conf = y_pred_proba[y_test == 0]
    
    confidence_metrics['mean_confidence_positive_class'] = np.mean(positive_class_conf) if len(positive_class_conf) > 0 else np.nan
    confidence_metrics['mean_confidence_negative_class'] = np.mean(negative_class_conf) if len(negative_class_conf) > 0 else np.nan
    confidence_metrics['confidence_separation'] = confidence_metrics['mean_confidence_positive_class'] - confidence_metrics['mean_confidence_negative_class']
    
    # Confidence distribution analysis
    high_conf_threshold = 0.8
    low_conf_threshold = 0.2
    
    high_conf_predictions = np.sum((y_pred_proba > high_conf_threshold) | (y_pred_proba < low_conf_threshold))
    uncertain_predictions = np.sum((y_pred_proba >= low_conf_threshold) & (y_pred_proba <= high_conf_threshold))
    
    confidence_metrics['high_confidence_ratio'] = high_conf_predictions / len(y_pred_proba)
    confidence_metrics['uncertain_predictions_ratio'] = uncertain_predictions / len(y_pred_proba)
    confidence_metrics['confidence_std'] = np.std(y_pred_proba)
    
    return confidence_metrics

def extract_threshold_metrics(y_test, y_pred_proba):
    """Analyze performance at different decision thresholds"""
    threshold_metrics = {}
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        
        acc = accuracy_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        
        threshold_metrics[f'accuracy_thresh_{thresh}'] = acc
        threshold_metrics[f'f1_thresh_{thresh}'] = f1
        threshold_metrics[f'precision_thresh_{thresh}'] = prec
        threshold_metrics[f'recall_thresh_{thresh}'] = rec
    
    return threshold_metrics

def evaluate_comprehensive_metrics(model, X_data, y_data, data_type="test"):
    """Evaluate all metrics on given dataset"""
    # Get predictions
    y_pred_proba = model.predict(X_data, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate loss
    loss = model.evaluate(X_data, y_data, verbose=0)
    if isinstance(loss, list):
        loss = loss[0]  # Get loss value if multiple metrics returned
    
    # Standard metrics
    metrics = {
        f"{data_type}_loss": loss,
        f"{data_type}_accuracy": accuracy_score(y_data, y_pred),
        f"{data_type}_f1": f1_score(y_data, y_pred, zero_division=0),
        f"{data_type}_precision": precision_score(y_data, y_pred, zero_division=0),
        f"{data_type}_recall": recall_score(y_data, y_pred, zero_division=0),
        f"{data_type}_auc": roc_auc_score(y_data, y_pred_proba) if len(np.unique(y_data)) == 2 else np.nan,
        f"{data_type}_bal_acc": balanced_accuracy_score(y_data, y_pred),
    }
    
    # Confidence metrics
    confidence_metrics = extract_confidence_metrics(y_data, y_pred_proba)
    for key, value in confidence_metrics.items():
        metrics[f"{data_type}_{key}"] = value
    
    # Threshold metrics
    threshold_metrics = extract_threshold_metrics(y_data, y_pred_proba)
    for key, value in threshold_metrics.items():
        metrics[f"{data_type}_{key}"] = value
    
    return metrics

def evaluate_lstm_model_comprehensive(model, history, X_train, y_train, X_test, y_test, test_starts, test_session_ids, repeat, fold):
    """Comprehensive evaluation on both train and test sets"""
    
    # Get training history metrics
    training_history = {
        'final_train_loss_history': history.history['loss'][-1],
        'epochs_trained': len(history.history['loss']),
    }
    
    # Evaluate on training set
    print(f"\nEvaluating on training set...")
    train_metrics = evaluate_comprehensive_metrics(model, X_train, y_train, "train")
    
    # Evaluate on test set
    print(f"Evaluating on test set...")
    test_metrics = evaluate_comprehensive_metrics(model, X_test, y_test, "test")
    
    # Combine all metrics
    all_metrics = {}
    all_metrics.update(training_history)
    all_metrics.update(train_metrics)
    all_metrics.update(test_metrics)
    
    # Calculate differences between train and test
    comparison_metrics = {
        'train_test_loss_diff': train_metrics['train_loss'] - test_metrics['test_loss'],
        'train_test_acc_diff': train_metrics['train_accuracy'] - test_metrics['test_accuracy'],
        'train_test_auc_diff': train_metrics['train_auc'] - test_metrics['test_auc'],
        'train_test_f1_diff': train_metrics['train_f1'] - test_metrics['test_f1'],
    }
    all_metrics.update(comparison_metrics)
    
    # Create test results dataframe for temporal analysis
    y_pred_proba_test = model.predict(X_test, verbose=0).flatten()
    y_pred_test = (y_pred_proba_test >= 0.5).astype(int)
    
    results_df = pd.DataFrame({
        'session': test_session_ids,
        'start_frame': test_starts,
        'true_label': y_test,
        'pred_label': y_pred_test,
        'pred_proba': y_pred_proba_test
    })
    
    # Extract temporal metrics (only for test set)
    temporal_metrics = extract_temporal_metrics(results_df)
    all_metrics.update(temporal_metrics)
    
    # Print comprehensive results
    print(f"\nLSTM Results (Repeat {repeat+1}, Fold {fold+1}):")
    print(f"  Test Samples: {len(y_test)} | Train Samples: {len(y_train)}")
    print(f"  Test Predicted positive: {np.sum(y_pred_test)} | True positive: {np.sum(y_test)}")
    
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  {'Metric':<15} {'Train':<8} {'Test':<8} {'Diff':<8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Loss':<15} {train_metrics['train_loss']:<8.4f} {test_metrics['test_loss']:<8.4f} {comparison_metrics['train_test_loss_diff']:<8.4f}")
    print(f"  {'Accuracy':<15} {train_metrics['train_accuracy']:<8.4f} {test_metrics['test_accuracy']:<8.4f} {comparison_metrics['train_test_acc_diff']:<8.4f}")
    print(f"  {'F1':<15} {train_metrics['train_f1']:<8.4f} {test_metrics['test_f1']:<8.4f} {comparison_metrics['train_test_f1_diff']:<8.4f}")
    print(f"  {'AUC':<15} {train_metrics['train_auc']:<8.4f} {test_metrics['test_auc']:<8.4f} {comparison_metrics['train_test_auc_diff']:<8.4f}")
    
    # Overfitting analysis
    print(f"\nOVERFITTING ANALYSIS:")
    if comparison_metrics['train_test_loss_diff'] < -0.1:
        print("  → Significant overfitting detected (train loss << test loss)")
    elif comparison_metrics['train_test_acc_diff'] > 0.1:
        print("  → Overfitting detected (train accuracy >> test accuracy)")
    elif comparison_metrics['train_test_loss_diff'] > 0.1:
        print("  → Possible underfitting (train loss >> test loss)")
    else:
        print("  → Training appears balanced")
    
    print(f"  Epochs Trained: {all_metrics['epochs_trained']}")
    
    # Temporal analysis for test set
    if REAL_TIME_MODE:
        print(f"\nREAL-TIME DETECTION ANALYSIS (Test Set):")
        
        for session in np.unique(test_session_ids):
            session_data = results_df[results_df['session'] == session].sort_values('start_frame')
            true_label = session_data['true_label'].iloc[0]
            pred_positive_windows = np.sum(session_data['pred_label'])
            total_windows = len(session_data)
            
            if true_label == 1:  # Exclusion session
                first_detection_frame = session_data[session_data['pred_label'] == 1]['start_frame'].min()
                detection_rate = pred_positive_windows / total_windows
                print(f"    {session} (exclusion): {pred_positive_windows}/{total_windows} windows detected ({detection_rate:.2f})")
                if not pd.isna(first_detection_frame):
                    delay_sec = first_detection_frame / 30
                    print(f"      First detection at frame {first_detection_frame} ({delay_sec:.1f}s)")
                else:
                    print(f"      No detection")
            else:  # Normal session
                false_alarm_rate = pred_positive_windows / total_windows
                print(f"    {session} (normal): {pred_positive_windows}/{total_windows} false alarms ({false_alarm_rate:.2f})")
    
    return all_metrics

def save_comprehensive_results(run_metrics, results_df):
    """Save comprehensive results including train/test comparisons"""
    
    # Create detailed fold results
    detailed_results = []
    
    for idx, row in results_df.iterrows():
        fold_info = {
            'repeat': row['repeat'],
            'fold': row['fold'],
            'run_id': f"repeat_{row['repeat']}_fold_{row['fold']}",
            'sequence_length': row['sequence_length'],
            'step_size': row['step_size'],
            'features_used': row['features_used'],
            'train_samples': row['train_samples'],
            'test_samples': row['test_samples'],
            'epochs_trained': row.get('epochs_trained', np.nan),
            
            # Training metrics
            'train_loss': row.get('train_loss', np.nan),
            'train_accuracy': row.get('train_accuracy', np.nan),
            'train_f1': row.get('train_f1', np.nan),
            'train_precision': row.get('train_precision', np.nan),
            'train_recall': row.get('train_recall', np.nan),
            'train_auc': row.get('train_auc', np.nan),
            
            # Test metrics
            'test_loss': row.get('test_loss', np.nan),
            'test_accuracy': row.get('test_accuracy', np.nan),
            'test_f1': row.get('test_f1', np.nan),
            'test_precision': row.get('test_precision', np.nan),
            'test_recall': row.get('test_recall', np.nan),
            'test_auc': row.get('test_auc', np.nan),
            
            # Comparison metrics
            'train_test_loss_diff': row.get('train_test_loss_diff', np.nan),
            'train_test_acc_diff': row.get('train_test_acc_diff', np.nan),
            'train_test_auc_diff': row.get('train_test_auc_diff', np.nan),
            'train_test_f1_diff': row.get('train_test_f1_diff', np.nan),
            
            # Temporal metrics
            'mean_detection_delay_seconds': row.get('mean_detection_delay_seconds', np.nan),
            'detection_success_rate': row.get('detection_success_rate', np.nan),
            'mean_session_detection_rate': row.get('mean_session_detection_rate', np.nan),
            
            # Confidence metrics
            'test_confidence_separation': row.get('test_confidence_separation', np.nan),
            'test_high_confidence_ratio': row.get('test_high_confidence_ratio', np.nan),
            'test_uncertain_predictions_ratio': row.get('test_uncertain_predictions_ratio', np.nan),
        }
        detailed_results.append(fold_info)
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    mode_suffix = "_comprehensive_lstm"
    detailed_df.to_csv(f"{FOLDER_NAME}/comprehensive_results{mode_suffix}.csv", index=False)
    
    print(f"\nComprehensive results saved:")
    print(f"  - {FOLDER_NAME}/comprehensive_results{mode_suffix}.csv")
    
    return detailed_df

def print_comprehensive_summary(results_df):
    """Print comprehensive summary of train vs test performance"""
    print("\n=== COMPREHENSIVE LSTM RESULTS SUMMARY ===")
    
    # Test performance
    test_metrics = ["test_accuracy", "test_f1", "test_auc", "test_precision", "test_recall"]
    print("\nTEST SET PERFORMANCE:")
    for metric in test_metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Training performance
    train_metrics = ["train_accuracy", "train_f1", "train_auc", "train_precision", "train_recall"]
    print("\nTRAINING SET PERFORMANCE:")
    for metric in train_metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Loss analysis
    loss_metrics = ["train_loss", "test_loss", "train_test_loss_diff"]
    print("\nLOSS ANALYSIS:")
    for metric in loss_metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Overfitting analysis
    if 'train_test_loss_diff' in results_df.columns and 'train_test_acc_diff' in results_df.columns:
        loss_diff = results_df['train_test_loss_diff'].mean()
        acc_diff = results_df['train_test_acc_diff'].mean()
        
        print(f"\nOVERFITTING ANALYSIS:")
        print(f"  Train-Test Loss Difference: {loss_diff:.4f}")
        print(f"  Train-Test Accuracy Difference: {acc_diff:.4f}")
        
        if loss_diff < -0.1 or acc_diff > 0.1:
            print("  → Overfitting detected - train performance much better than test")
        elif loss_diff > 0.1 or acc_diff < -0.1:
            print("  → Underfitting detected - test performance better than train")
        else:
            print("  → Training appears well-balanced")

# Updated main run function
def run_experiment():
    """Main experiment runner with comprehensive metrics"""
    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)
        
    session_files = list(labels.keys())
    session_labels = list(labels.values())
    
    os.makedirs(FOLDER_NAME, exist_ok=True)
    run_metrics = []
    all_predictions = []

    # Print data summary
    print(f"\nDATA SUMMARY:")
    print(f"  Total sessions: {len(session_files)}")
    print(f"  Exclusion sessions: {sum(session_labels)} ({sum(session_labels)/len(session_labels)*100:.1f}%)")
    print(f"  Normal sessions: {len(session_labels) - sum(session_labels)} ({(len(session_labels) - sum(session_labels))/len(session_labels)*100:.1f}%)")

    for repeat in range(REPEATS):
        print(f"\nRepetition {repeat + 1}/{REPEATS}")
        
        # np.random.seed(SEED + repeat)
        # tf.random.set_seed(SEED + repeat)
        
        # Split into train and test
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(skf.split(session_files, session_labels)):
            print(f"— Fold {fold + 1}")
            
            # Get train and test files
            train_files = [session_files[i] for i in train_idx]
            test_files = [session_files[i] for i in test_idx]
            
            train_labels = {f: labels[f] for f in train_files}
            test_labels = {f: labels[f] for f in test_files}
            
            # Print fold composition
            train_exclusions = sum(train_labels.values())
            test_exclusions = sum(test_labels.values())
            print(f"    Train: {len(train_files)} sessions ({train_exclusions} exclusions)")
            print(f"    Test: {len(test_files)} sessions ({test_exclusions} exclusions)")
            
            # Feature selection
            train_feats_all = get_feature_names_for_split(DATA_DIR, train_files, EXPERIMENT_GROUP)
            top_feature_names = select_features_from_json(train_feats_all, FEATURES_JSON)
            
            print(f"Using {len(top_feature_names)} features for experiment group '{EXPERIMENT_GROUP}'")
            
            # Fit scaler and load data
            scaler = fit_global_scaler(DATA_DIR, train_labels, selected_features=top_feature_names)
            
            X_train, y_train, _, train_session_ids = load_sessions(
                DATA_DIR, train_labels, scaler, SEQUENCE_LENGTH, STEP_SIZE, top_feature_names
            )
            X_test, y_test, starts, test_session_ids = load_sessions(
                DATA_DIR, test_labels, scaler, SEQUENCE_LENGTH, STEP_SIZE, top_feature_names
            )

            print(f"Train: {X_train.shape} | Test: {X_test.shape}")
            print(f"Train classes: {np.bincount(y_train)} | Test classes: {np.bincount(y_test)}")
            
            # Train LSTM model
            model, history = train_lstm_model(X_train, X_test, y_train, y_test)
            
            # Comprehensive evaluation
            test_metrics = evaluate_lstm_model_comprehensive(
                model, history, X_train, y_train, X_test, y_test, starts, test_session_ids, repeat, fold
            )
            
            # Store individual fold predictions for later analysis
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            for i, session_id in enumerate(test_session_ids):
                prediction_row = {
                    'repeat': repeat,
                    'fold': fold,
                    'session_id': session_id,
                    'start_frame': starts[i],
                    'true_label': y_test[i],
                    'pred_label': y_pred[i],
                    'pred_proba': y_pred_proba[i],
                    'session_file': [f for f in test_files if f.startswith(session_id)][0] if any(f.startswith(session_id) for f in test_files) else 'unknown'
                }
                all_predictions.append(prediction_row)
            
            # Store results
            training_metrics = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(top_feature_names),
                'sequence_length': SEQUENCE_LENGTH,
                'step_size': STEP_SIZE,
                'real_time_mode': REAL_TIME_MODE,
                'train_files': ','.join(train_files),
                'test_files': ','.join(test_files)
            }
            
            run_metrics.append({
                "repeat": repeat,
                "fold": fold,
                **training_metrics,
                **test_metrics
            })
    
    # Save all results
    results_df = pd.DataFrame(run_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    
    # Save comprehensive results
    comprehensive_df = save_comprehensive_results(run_metrics, results_df)
    
    # Save session predictions
    predictions_df.to_csv(f"{FOLDER_NAME}/session_predictions_comprehensive.csv", index=False)
    print(f"  - {FOLDER_NAME}/session_predictions_comprehensive.csv")

    print_comprehensive_summary(results_df)
    
    return results_df

if __name__ == '__main__':
    print("LSTM Experiment for Social Exclusion Detection")
    
    results = run_experiment()
    
    print("="*50)
    print("Configuration:")
    print(f"  - Sequence Length: {SEQUENCE_LENGTH} frames (60 seconds)")
    print(f"  - Step Size: {STEP_SIZE} frames (50% overlap)")
    print(f"  - Repeats: {REPEATS}")
    print(f"  - Folds: {N_SPLITS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print("="*50)
    print("\nLSTM Experiment completed!")