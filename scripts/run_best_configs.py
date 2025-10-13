import os
import re
import seaborn as sns
import json
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.pipeline import Pipeline

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ===============================
# USER SWITCHES
# ===============================
# Choose which dataset to run:
# - "participant" → per-participant labels (e.g., unreliable_labels.json)
# - "session"     → per-session labels (e.g., sessions_labels.json)
LEVEL = "session"  # <-- change to "session" when needed

# Pathse
DATA_DIR = "/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/pose_metrics_dynamics/relative/"
FOLDER_NAME = f"best_model_results-truncated-{LEVEL}"

# Labels file by level
LABELS_FILE_PARTICIPANT = "unreliable_labels.json"
LABELS_FILE_SESSION     = "session_labels.json"

# CV config
N_SPLITS   = 3   # folds for StratifiedGroupKFold
N_REPEATS  = 10  # repeat the whole CV with different seeds (stability)

# Randomness
SEED = 42
os.makedirs(FOLDER_NAME, exist_ok=True)

# Colors for feature categories
CATEGORY_COLORS = {
    "Motion": "#4c78a8",         # blue
    "Interpersonal": "#59a14f",  # green
    "AU": "#f28e2b"              # orange
}

# ===============================
# CONFIG GRID
# ===============================
# Session-level
BEST_CONFIG = {
    'name': 'more_features__motion__k600',
    'feature_selection': 'f_classif',
    'scaler': 'robust',
    'include_interactions': False,
    'include_temporal_features': True,
    'modality': 'motion',
    'k_features': 600,
    'model': 'LogisticRegression_L1'
}

# Participant-level
# BEST_CONFIG = {
#     'name': 'baseline__all__k20',
#     'feature_selection': 'f_classif',
#     'scaler': 'robust',
#     'include_interactions': False,
#     'include_temporal_features': False,
#     'modality': 'all',
#     'k_features': 20,
#     'model': 'SVM_Linear'
# }

GLOBAL_SELECTION_COUNTS = defaultdict(int)
GLOBAL_FEATURE_WEIGHTS = defaultdict(list)

STAT_SUFFIXES = [
    '_mean', '_std', '_median', '_iqr', '_mad', '_skew', '_kurtosis',
    '_cv', '_high_activity_ratio', '_low_activity_ratio',
    '_change_mean', '_change_std', '_trend_slope', '_trend_r2',
    '_autocorr_lag1', '_change'
]


# ===============================
# UTIL: Column detection helpers
# ===============================
def trim_raw_feature_name(feat):
    for suffix in sorted(STAT_SUFFIXES, key=len, reverse=True):  # longest first
        if feat.endswith(suffix):
            feat = feat[:-len(suffix)]
            break
    if feat.startswith("pair_mean__"):
        feat = feat[len("pair_mean__"):]
    elif feat.startswith("pair_abs__"):
        feat = feat[len("pair_abs__"):]
    return feat

def _is_au(col: str) -> bool:
    return ('AU' in col) and col.endswith('_r')

def _is_motion(col: str) -> bool:
    c = col.lower()
    return any(x in c for x in [
        'disp_', 'vel_', 'drift_', 'head_', 'shoulder_', 'hand_', 'reach_', 'torso_', 'hip_', 'elbow_', 'wrist_',
        'x_', 'y_', 'z_',
    ]) and not _is_au(col)

def _is_pair(col: str) -> bool:
    c = col.lower()
    return any(x in c for x in ['pair_', 'rel_', 'dist_', 'inter_', 'between_', 'gap_', 'proximity_'])

def _participant_prefixes(cols):
    prefixes = []
    p1 = [c for c in cols if c.startswith('p1_')]
    p2 = [c for c in cols if c.startswith('p2_')]
    A  = [c for c in cols if c.startswith('A_')]
    B  = [c for c in cols if c.startswith('B_')]
    if p1 and p2:
        prefixes = ['p1_', 'p2_']
    elif A and B:
        prefixes = ['A_', 'B_']
    return prefixes

def _axes_in_name(col):
    m = re.search(r'([xyz])_(\d+)', col)
    if m:
        return m.group(1), int(m.group(2))
    return None, None
    
def extract_model_weights(model):
    """Extract weights from model, handling both Pipeline and direct models."""
    if hasattr(model, 'coef_'):
        # Direct model (not in pipeline)
        return model.coef_[0]
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf'), 'coef_'):
        # Pipeline with 'clf' step
        return model.named_steps['clf'].coef_[0]
    elif hasattr(model, '_final_estimator') and hasattr(model._final_estimator, 'coef_'):
        # Pipeline with final estimator
        return model._final_estimator.coef_[0]
    else:
        return None
    
    
def count_raw_feature_usage(selected_features, output_path):
    """
    Count how often each raw feature was selected across folds/repeats.
    Examples:
    - 'disp_5_mean' -> 'disp_5' (displacement of keypoint 5)
    - 'vel_12_std' -> 'vel_12'
    - 'AU01_r_median' -> 'AU01_r'
    - 'shoulder_yaw_skew' -> 'shoulder_yaw'
    """

    stat_suffixes = [
        '_mean', '_std', '_median', '_iqr', '_mad', '_skew', '_kurtosis',
        '_cv', '_high_activity_ratio', '_low_activity_ratio',
        '_change_mean', '_change_std', '_trend_slope', '_trend_r2', '_autocorr_lag1'
    ]

    counts = Counter()
    for feat in selected_features:
        raw_name = feat
        for suffix in stat_suffixes:
            if raw_name.endswith(suffix):
                raw_name = raw_name[:-len(suffix)]
                break
        if raw_name.startswith('pair_mean__'):
            raw_name = raw_name[len('pair_mean__'):]
        elif raw_name.startswith('pair_abs__'):
            raw_name = raw_name[len('pair_abs__'):]

        counts[raw_name] += 1

    df_counts = pd.DataFrame({
        'raw_feature': list(counts.keys()),
        'selection_count': list(counts.values())
    }).sort_values("selection_count", ascending=False)

    df_counts.to_csv(output_path, index=False)
    print(f"[✅ raw feature usage] Saved to: {output_path}")
    return df_counts

# ===============================
# FEATURE ENGINEERING
# ===============================
def _basic_stats_for_series(name, data, out):
    mu = float(np.mean(data))
    sd = float(np.std(data))
    out[f'{name}_mean']   = mu
    out[f'{name}_std']    = sd
    out[f'{name}_median'] = float(np.median(data))
    out[f'{name}_iqr']    = float(np.percentile(data, 75) - np.percentile(data, 25))
    out[f'{name}_mad']    = float(np.median(np.abs(data - np.median(data))))
    out[f'{name}_skew']   = float(stats.skew(data)) if len(data) > 3 else 0.0
    out[f'{name}_kurtosis'] = float(stats.kurtosis(data)) if len(data) > 3 else 0.0
    if sd > 1e-6:
        th = mu + 1.5 * sd
        tl = mu - 1.5 * sd
        out[f'{name}_high_activity_ratio'] = float(np.sum(data > th) / len(data))
        out[f'{name}_low_activity_ratio']  = float(np.sum(data < tl) / len(data))
        out[f'{name}_cv'] = float(sd / (abs(mu) + 1e-8))
    else:
        out[f'{name}_high_activity_ratio'] = 0.0
        out[f'{name}_low_activity_ratio']  = 0.0
        out[f'{name}_cv'] = 0.0

def _temporal_stats_for_series(name, data, out):
    if len(data) <= 10:
        out[f'{name}_change_mean'] = 0.0
        out[f'{name}_change_std']  = 0.0
        out[f'{name}_trend_slope'] = 0.0
        out[f'{name}_trend_r2']    = 0.0
        out[f'{name}_autocorr_lag1'] = 0.0
        return
    diffd = np.diff(data)
    out[f'{name}_change_mean'] = float(np.mean(diffd)) if len(diffd) > 0 else 0.0
    out[f'{name}_change_std']  = float(np.std(diffd))  if len(diffd) > 0 else 0.0
    x = np.arange(len(data))
    if np.std(data) > 1e-8:
        slope, _, r_value, _, _ = stats.linregress(x, data)
        out[f'{name}_trend_slope'] = float(slope)
        out[f'{name}_trend_r2']    = float(r_value ** 2)
    else:
        out[f'{name}_trend_slope'] = 0.0
        out[f'{name}_trend_r2']    = 0.0
    if len(data) > 1 and np.std(data) > 1e-8:
        out[f'{name}_autocorr_lag1'] = float(np.corrcoef(data[:-1], data[1:])[0, 1])
    else:
        out[f'{name}_autocorr_lag1'] = 0.0

def _add_series_feature_block(name, data, out, include_temporal):
    _basic_stats_for_series(name, data, out)
    if include_temporal:
        _temporal_stats_for_series(name, data, out)

def _interpersonal_block_from_coords(df, include_temporal, out_prefix="ip_"):
    cols = df.columns.tolist()
    pfx = _participant_prefixes(cols)
    if not pfx:
        return {}
    pA, pB = pfx[0], pfx[1]

    joints = {}
    for c in cols:
        if c.startswith(pA) or c.startswith(pB):
            axis, j = _axes_in_name(c[len(pA):] if c.startswith(pA) else c[len(pB):])
            if axis is None:
                continue
            key = (j)
            if key not in joints:
                joints[key] = {'A': {}, 'B': {}}
            if c.startswith(pA):
                joints[key]['A'][axis] = c
            else:
                joints[key]['B'][axis] = c

    features = {}
    for j, sides in joints.items():
        A, B = sides['A'], sides['B']
        if 'x' in A and 'x' in B and 'y' in A and 'y' in B and 'z' in A and 'z' in B:
            v = np.sqrt((df[A['x']]-df[B['x']])**2 + (df[A['y']]-df[B['y']])**2 + (df[A['z']]-df[B['z']])**2).values
            _add_series_feature_block(f'{out_prefix}dist_joint{j}', v, features, include_temporal)
        if 'x' in A and 'x' in B:
            v = np.abs(df[A['x']].values - df[B['x']].values)
            _add_series_feature_block(f'{out_prefix}dx_abs_joint{j}', v, features, include_temporal)
        if 'z' in A and 'z' in B:
            v = np.abs(df[A['z']].values - df[B['z']].values)
            _add_series_feature_block(f'{out_prefix}dz_abs_joint{j}', v, features, include_temporal)

    return features

def extract_enhanced_features(
    df,
    include_interactions=False,
    include_temporal_features=False,
    modality='all',
    level='participant'
):
    """
    Enhanced feature extraction with:
    - modality filtering: 'all' | 'motion' | 'au' | 'interpersonal'
    - session-level interpersonal derivations from p1/p2 or A/B coordinate columns
    - inclusion of any prebuilt pair/relative columns (pair_*, rel_*, dist_*) as interpersonal
    """
    df = df.drop(columns=["frame", "window_id"], errors="ignore")
    df = df.select_dtypes(include=[np.number]).fillna(0.0)

    cols = df.columns.tolist()
    au_cols      = [c for c in cols if _is_au(c)]
    motion_cols  = [c for c in cols if _is_motion(c) and c not in au_cols]
    pair_cols    = [c for c in cols if _is_pair(c)]

    interpersonal_features = {}
    if level == 'session':
        if pair_cols:
            pass
        ip_from_coords = _interpersonal_block_from_coords(df, include_temporal_features, out_prefix="ip_")
        interpersonal_features.update(ip_from_coords)

    if modality == 'au':
        keep_cols = au_cols
    elif modality == 'motion':
        keep_cols = motion_cols
    elif modality == 'interpersonal':
        keep_cols = pair_cols
    else:
        keep_cols = list(cols)

    if len(keep_cols) == 0:
        print(f"[WARN] No raw columns for modality='{modality}'. Falling back to all numeric.")
        keep_cols = list(cols)

    features = {}
    for col in keep_cols:
        data = df[col].values
        _add_series_feature_block(col, data, features, include_temporal_features)

    if level == 'session' and interpersonal_features and (modality in ['all', 'interpersonal']):
        features.update(interpersonal_features)

    if include_interactions and modality == 'all' and au_cols and motion_cols:
        au_mean = np.mean(df[au_cols].values, axis=1) if au_cols else None
        mot_val = df[motion_cols].values
        motion_intensity = np.sqrt(np.sum(mot_val**2, axis=1)) if motion_cols else None
        if au_mean is not None and motion_intensity is not None:
            if np.std(au_mean) > 1e-8 and np.std(motion_intensity) > 1e-8:
                features['au_motion_correlation'] = float(np.corrcoef(au_mean, motion_intensity)[0, 1])
            else:
                features['au_motion_correlation'] = 0.0
            high_au = au_mean > np.percentile(au_mean, 75)
            high_mo = motion_intensity > np.percentile(motion_intensity, 75)
            features['high_au_high_motion_overlap'] = float(np.sum(high_au & high_mo) / len(au_mean))
            features['motion_during_high_au'] = float(np.mean(motion_intensity[high_au])) if np.sum(high_au) > 0 else 0.0
            features['au_during_high_motion'] = float(np.mean(au_mean[high_mo])) if np.sum(high_mo) > 0 else 0.0

    return features

def extract_raw_feature_weights(selected_features, weights, output_path):
    """
    Groups engineered features back to their raw base features with better parsing.
    Examples:
    - 'disp_5_mean' -> 'disp_5' (displacement of keypoint 5)
    - 'vel_12_std' -> 'vel_12' (velocity of keypoint 12)
    - 'AU01_r_median' -> 'AU01_r' (Action Unit 1)
    - 'shoulder_yaw_skew' -> 'shoulder_yaw'
    """
    raw_weights = defaultdict(float)
    
    # Statistical suffixes to remove
    stat_suffixes = [
        '_mean', '_std', '_median', '_iqr', '_mad', '_skew', '_kurtosis',
        '_cv', '_high_activity_ratio', '_low_activity_ratio',
        '_change_mean', '_change_std', '_trend_slope', '_trend_r2', '_autocorr_lag1'
    ]
    
    for feat, w in zip(selected_features, weights):
        raw_name = feat
        
        # Remove statistical suffixes
        for suffix in stat_suffixes:
            if raw_name.endswith(suffix):
                raw_name = raw_name[:-len(suffix)]
                break
        
        # Handle special cases
        if raw_name.startswith('pair_mean__'):
            raw_name = raw_name[len('pair_mean__'):]
        elif raw_name.startswith('pair_abs__'):
            raw_name = raw_name[len('pair_abs__'):]
        elif '_' in raw_name and raw_name.split('_')[-1].isdigit():
            # Features like 'disp_5', 'vel_12' - keep as is
            pass
        elif raw_name.endswith('_r') and 'AU' in raw_name:
            # Action Units like 'AU01_r' - keep as is
            pass
        
        raw_weights[raw_name] += abs(w)
    
    # Create DataFrame and save
    df_raw = pd.DataFrame({
        "raw_feature": list(raw_weights.keys()),
        "aggregated_weight": list(raw_weights.values())
    }).sort_values("aggregated_weight", ascending=False)
    
    df_raw.to_csv(output_path, index=False)
    print(f"[✅ raw weights] Saved to: {output_path}")
    
    return df_raw

def analyze_raw_features_by_type(csv_path, output_dir):
    """
    Analyzes raw features by type (motion, facial, interpersonal) and creates separate plots.
    """
    df = pd.read_csv(csv_path)
    
    # Categorize features
    motion_features = []
    facial_features = []
    interpersonal_features = []
    other_features = []
    
    for _, row in df.iterrows():
        feat = row['raw_feature']
        weight = row.get('aggregated_weight', row.get('selection_count', 0))
        
        # For session-level features, look AFTER the pair_ prefix
        if feat.startswith('pair_'):
            # Extract the part after pair_mean__ or pair_abs__
            if '__' in feat:
                core_feature = feat.split('__', 1)[1]  # Get everything after first __
            else:
                core_feature = feat
            
            # Now categorize based on the core feature
            if any(x in core_feature.lower() for x in ['disp_', 'vel_', 'drift_', 'head_', 'shoulder_', 'hand_']):
                motion_features.append((feat, weight))
            elif core_feature.startswith('AU') and core_feature.endswith('_r'):
                facial_features.append((feat, weight))
            else:
                other_features.append((feat, weight))
        else:
            # Non-pair features (shouldn't exist in session-level, but just in case)
            if any(x in feat.lower() for x in ['disp_', 'vel_', 'drift_', 'head_', 'shoulder_', 'hand_']):
                motion_features.append((feat, weight))
            elif feat.startswith('AU') and feat.endswith('_r'):
                facial_features.append((feat, weight))
            else:
                other_features.append((feat, weight))
    
    # Create summary plot
    categories = ['Motion', 'Facial', 'Interpersonal', 'Other']
    total_weights = [
        sum(w for _, w in motion_features),
        sum(w for _, w in facial_features), 
        sum(w for _, w in interpersonal_features),
        sum(w for _, w in other_features)
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, total_weights, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels
    for bar, weight in zip(bars, total_weights):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(total_weights),
                f'{weight:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Total Feature Importance by Category', fontsize=14, fontweight='bold')
    plt.ylabel('Total Aggregated Weight', fontsize=12)
    plt.xlabel('Feature Category', fontsize=12)
    
    summary_path = f"{output_dir}/raw_features_by_category.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✅ plot] Saved category summary to: {summary_path}")
    
    return {
        'motion': motion_features,
        'facial': facial_features, 
        'interpersonal': interpersonal_features,
        'other': other_features
    }

# ===============================
# DATA LOADING
# ===============================
def get_session_groups_for_participants(participant_ids):
    return [pid.split('_')[0] for pid in participant_ids]

def _labels_file_for_level():
    return LABELS_FILE_SESSION if LEVEL == "session" else LABELS_FILE_PARTICIPANT

def load_and_engineer_features(data_dir, labels_dict, config):
    X_rows, y_list, example_ids, failed = [], [], [], []

    for filename, label in labels_dict.items():
        try:
            df = pd.read_csv(os.path.join(data_dir, filename))
            feats = extract_enhanced_features(
                df,
                include_interactions=config['include_interactions'],
                include_temporal_features=config['include_temporal_features'],
                modality=config.get('modality', 'all'),
                level=LEVEL
            )
            X_rows.append(list(feats.values()))
            y_list.append(label)
            example_ids.append(filename)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            failed.append(filename)

    if not X_rows:
        raise ValueError("No data could be loaded!")

    first_file = None
    for fn in labels_dict.keys():
        try:
            df0 = pd.read_csv(os.path.join(data_dir, fn))
            feats0 = extract_enhanced_features(
                df0,
                include_interactions=config['include_interactions'],
                include_temporal_features=config['include_temporal_features'],
                modality=config.get('modality', 'all'),
                level=LEVEL
            )
            first_file = fn
            break
        except Exception:
            continue
    if first_file is None:
        raise RuntimeError("Could not build feature names from any file.")

    feature_names = list(feats0.keys())
    X = np.array(X_rows, dtype=float)
    y = np.array(y_list)

    print(f"Engineered features shape: {X.shape}  |  Failed loads: {len(failed)}")
    return X, y, example_ids, feature_names

# ===============================
# FEATURE SELECTION / SCALERS / MODELS
# ===============================
def get_feature_selector(method, k, seed):
    if method == 'f_classif':
        return SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        return SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
        return RFE(estimator, n_features_to_select=k)
    else:
        return SelectKBest(score_func=f_classif, k=k)

def get_scaler(scaler_type):
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'robust':
        return RobustScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    else:
        return RobustScaler()

# --- Feature selection logs (across repeats & folds) ---
FEATURE_LOGS = []  # one row per selected feature per (repeat, fold)
RUN_LOGS = []      # denominator: one row per (repeat, fold) run

def _register_selection_usage(sel_detail_rows, context):
    """
    sel_detail_rows: list of dicts with {'feature': str, 'score': float (optional)}
    context: dict with keys like level/base_config/modality/k_requested/repeat/fold
    """
    for row in sel_detail_rows:
        entry = dict(context)
        entry.update({
            'feature': row.get('feature'),
            'score': row.get('score', np.nan)
        })
        FEATURE_LOGS.append(entry)

def select_best_features(X_train, y_train, feature_names, config, seed):
    requested_k = config['k_features']
    method = config['feature_selection']
    print(f"\n=== FEATURE SELECTION ({method}, k={requested_k}) ===")

    # Remove zero-variance features
    non_zero_var = np.var(X_train, axis=0) > 1e-8
    X_train_filtered = X_train[:, non_zero_var]
    filtered_names = [feature_names[i] for i in range(len(feature_names)) if non_zero_var[i]]
    print(f"Removed {np.sum(~non_zero_var)} zero-variance features")

    # Resolve k
    if requested_k == "all":
        k = X_train_filtered.shape[1]
    else:
        k = int(requested_k)
    if X_train_filtered.shape[1] < k:
        k = X_train_filtered.shape[1]
        print(f"Reducing k to {k} due to limited features")
    if k <= 0:
        k = X_train_filtered.shape[1]

    selector = get_feature_selector(method, k, seed)
    selector.fit(X_train_filtered, y_train)

    # Capture selected features (+ scores if available)
    sel_detail = []
    if hasattr(selector, 'get_support'):
        selected_indices = selector.get_support(indices=True)
        selected_features = [filtered_names[i] for i in selected_indices]
        if hasattr(selector, 'scores_') and selector.scores_ is not None:
            scores = selector.scores_[selected_indices]
            print("Selected features with scores (top-K):")
            for feat, score in zip(selected_features, scores):
                print(f"  {feat:<70} {score:.3f}")
                sel_detail.append({'feature': feat, 'score': float(score)})
        else:
            print("Selected feature names (no scores available):")
            for feat in selected_features:
                print(f"  {feat}")
                sel_detail.append({'feature': feat, 'score': np.nan})

    effective_k = k
    return selector, filtered_names, non_zero_var, effective_k, requested_k, sel_detail

def create_enhanced_models(config, seed):
    scaler = get_scaler(config['scaler'])
    models = {
        'RandomForest_Default': RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_split=3, min_samples_leaf=1,
            class_weight='balanced', random_state=seed, n_jobs=1
        ),
        'RandomForest_Conservative': RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', random_state=seed, n_jobs=1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, max_depth=4, min_samples_split=3, min_samples_leaf=1,
            class_weight='balanced', random_state=seed, n_jobs=1
        ),
        'LogisticRegression_L1': Pipeline([
            ('scaler', scaler),
            ('clf', LogisticRegression(
                penalty='l1', C=0.5, class_weight='balanced',
                solver='liblinear', max_iter=1000, random_state=seed
            ))
        ]),
        'LogisticRegression_L2': Pipeline([
            ('scaler', scaler),
            ('clf', LogisticRegression(
                penalty='l2', C=1.0, class_weight='balanced',
                solver='liblinear', max_iter=1000, random_state=seed
            ))
        ]),
        'SVM_Linear': Pipeline([
            ('scaler', scaler),
            ('clf', SVC(
                kernel='linear', C=0.5, class_weight='balanced',
                probability=True, random_state=seed
            ))
        ]),
        'SVM_RBF': Pipeline([
            ('scaler', scaler),
            ('clf', SVC(
                kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=seed
            ))
        ]),
        'Ridge': Pipeline([
            ('scaler', scaler),
            ('clf', RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=seed))
        ])
    }
    if not config['name'].startswith('baseline'):
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=seed)
        et = ExtraTreesClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=seed)
        models['Ensemble_RF_ET'] = VotingClassifier([('rf', rf), ('et', et)], voting='soft')
    return models

# ===============================
# EVALUATION
# ===============================
def evaluate_fold(X_train, X_test, y_train, y_test, feature_names, fold_idx, fold_info, config, seed, repeat_idx):
    print(f"\n--- {fold_info} :: {config['name']} (level={LEVEL}, mod={config.get('modality','all')}, k={config['k_features']}) ---")
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test  dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    selector, filtered_names, non_zero_mask, k_eff, k_req, sel_detail = select_best_features(
        X_train, y_train, feature_names, config, seed
    )

    # log denominator for feature usage
    RUN_LOGS.append({
        'level': LEVEL,
        'base_config': config['name'].split('__')[0],
        'modality': config.get('modality', 'all'),
        'k_requested': k_req,
        'repeat': repeat_idx,
        'fold': fold_idx
    })
    # log the selected features this run
    _register_selection_usage(sel_detail, {
        'level': LEVEL,
        'base_config': config['name'].split('__')[0],
        'modality': config.get('modality','all'),
        'k_requested': k_req,
        'repeat': repeat_idx,
        'fold': fold_idx
    })

    X_train_filtered = X_train[:, non_zero_mask]
    X_test_filtered  = X_test[:,  non_zero_mask]
    X_train_selected = selector.transform(X_train_filtered)
    X_test_selected  = selector.transform(X_test_filtered)

    print(f"Final feature space: {X_train_selected.shape[1]} features (requested={k_req}, effective={k_eff})")

    sel_idx = getattr(selector, "get_support", lambda **kw: None)(indices=True)
    selected_features = [filtered_names[i] for i in sel_idx] if sel_idx is not None else []

    models = create_enhanced_models(config, seed)
    target_model = config.get('model', 'LogisticRegression_L1')
    results = {}

    if target_model in models:
        model_name = target_model
        model = models[model_name]
        try:
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_selected)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test_selected)
                y_proba = 1 / (1 + np.exp(-y_scores))
            else:
                y_proba = None

            metrics = {
                'level': LEVEL,
                'config': config['name'],
                'base_config': config['name'].split('__')[0],
                'modality': config.get('modality', 'all'),
                'k_requested': k_req,
                'k_effective': int(X_train_selected.shape[1]),
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'n_features_used': int(X_train_selected.shape[1]),
                'selected_features': ";".join(selected_features),
                'repeat': repeat_idx,
                'fold': fold_idx
            }
            if y_proba is not None and len(np.unique(y_test)) == 2:
                metrics['auc'] = roc_auc_score(y_test, y_proba)
            else:
                metrics['auc'] = np.nan

            results[model_name] = metrics

            cm = confusion_matrix(y_test, y_pred)
            print(f"{model_name:>26} | BA={metrics['balanced_accuracy']:.3f}  F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  CM=[[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]")
            
            # Track only unique raw features per fold
            raw_feats_in_fold = set()
            for feat in selected_features:
                raw_feat = trim_raw_feature_name(feat)
                raw_feats_in_fold.add(raw_feat)

            for raw_feat in raw_feats_in_fold:
                GLOBAL_SELECTION_COUNTS[raw_feat] += 1
            
            final_model = model
            if isinstance(model, Pipeline):
                final_model = model.named_steps['clf']  # since your last step is named 'clf'

            if hasattr(final_model, "coef_"):
                weights = final_model.coef_[0]
                for feat, w in zip(selected_features, weights):
                    raw_feat = trim_raw_feature_name(feat)
                    GLOBAL_FEATURE_WEIGHTS[raw_feat].append(w)
                    
                                    
        except Exception as e:
            print(f"Model {model_name} failed: {e}")

    return results

def run_config_evaluation(config, X, y, groups, feature_names):
    print(f"\n{'='*90}")
    print(f"EVALUATING: {config['name']}  (level={LEVEL}, modality={config.get('modality','all')}, k={config['k_features']})")
    print(f"Selector: {config['feature_selection']} | Scaler: {config['scaler']}")
    print(f"Interactions: {config['include_interactions']} | Temporal: {config['include_temporal_features']}")
    print(f"{'='*90}")

    all_results = []

    for repeat_idx in range(N_REPEATS):
        local_seed = SEED + 1000 * repeat_idx
        print(f"\n>>> REPEAT {repeat_idx+1}/{N_REPEATS}  (seed={local_seed}) — Using {N_SPLITS}-Fold Stratified Group CV")

        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=local_seed)
        splits = list(cv.split(X, y, groups=groups))

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_test)) < 2:
                print(f"Skipping fold {fold_idx} - homogeneous test set")
                continue

            fold_results = evaluate_fold(
                X_train, X_test, y_train, y_test, feature_names,
                fold_idx,
                f"Repeat {repeat_idx+1} — Fold {fold_idx}/{len(splits)}",
                config,
                seed=local_seed,
                repeat_idx=repeat_idx
            )

            for _, metrics in fold_results.items():
                all_results.append(metrics)

    return all_results

# ===============================
# RUN + SAVE
# ===============================
def finalize_feature_selection_summary(output_dir):
    if not GLOBAL_SELECTION_COUNTS:
        print("[⚠️] No global selection counts found.")
        return
    df = pd.DataFrame({
        "raw_feature": list(GLOBAL_SELECTION_COUNTS.keys()),
        "selection_count": list(GLOBAL_SELECTION_COUNTS.values())
    })
    df = df.sort_values("selection_count", ascending=False)
    csv_path = os.path.join(output_dir, "raw_feature_selection_counts__all_folds.csv")
    df.to_csv(csv_path, index=False)
    print(f"[📄] Saved global selection counts to: {csv_path}")

    plot_path = os.path.join(output_dir, "feature_selection_plot__all_folds.png")
    plot_feature_selection_counts(GLOBAL_SELECTION_COUNTS, plot_path, top_n=None)
    if LEVEL == "session":
        plot_path = os.path.join(output_dir, "logreg_feature_weights.png")
        plot_feature_weights(GLOBAL_FEATURE_WEIGHTS, plot_path, top_n=None) 

def run_comprehensive_evaluation():
    print("Starting Comprehensive Model Evaluation")
    print("="*80)

    labels_file = _labels_file_for_level()
    with open(labels_file, "r") as f:
        labels = json.load(f)

    all_results = []
    config_summaries = []

    config = BEST_CONFIG
    X, y, example_ids, feature_names = load_and_engineer_features(DATA_DIR, labels, config)
    groups = get_session_groups_for_participants(example_ids)

    print(f"\nDataset summary: N={len(example_ids)} examples | Groups={len(set(groups))} | Features={len(feature_names)}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    config_results = run_config_evaluation(config, X, y, groups, feature_names)
    all_results.extend(config_results)

    if config_results:
        results_df = pd.DataFrame(config_results)
        best_models = results_df.groupby('model')['balanced_accuracy'].mean().sort_values(ascending=False)
        config_summary = {
            'level': LEVEL,
            'config': config['name'],
            'base_config': config['name'].split('__')[0],
            'modality': config.get('modality', 'all'),
            'k_requested': config['k_features'],
            'best_model': best_models.index[0],
            'best_bal_acc': float(best_models.iloc[0]),
            'n_models_tested': int(len(best_models)),
            'n_successful_folds': int(len(results_df) // len(best_models))
        }
        config_summaries.append(config_summary)

    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{FOLDER_NAME}/comprehensive_results.csv", index=False)

        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS SUMMARY")
        print(f"{'='*80}")

        summary_df = pd.DataFrame(config_summaries)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('best_bal_acc', ascending=False)
            print("\nBest model per configuration:")
            print(summary_df.to_string(index=False, float_format='%.3f'))
            summary_df.to_csv(f"{FOLDER_NAME}/config_summary.csv", index=False)

            overall_best_idx = results_df.groupby(['config', 'model'])['balanced_accuracy'].transform('mean').idxmax()
            overall_best = results_df.loc[overall_best_idx]
            print(f"\nOverall best combination:")
            print(f"  Level: {overall_best['level']}  |  Config: {overall_best['config']}  |  Model: {overall_best['model']}")
            print(f"  Balanced Accuracy: {overall_best['balanced_accuracy']:.3f}")

        model_summary = results_df.groupby(
            ['level', 'base_config', 'modality', 'k_requested','model']
        ).agg({
            'balanced_accuracy': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'auc': ['mean', 'std'],
            'k_effective': ['mean']
        }).round(3)
        model_summary.to_csv(f"{FOLDER_NAME}/detailed_model_summary.csv")

        finalize_feature_selection_summary(FOLDER_NAME)

        print(f"\nDetailed results saved to {FOLDER_NAME}/")
        return results_df, summary_df, model_summary
    else:
        print("No successful evaluations completed!")
        return None, None, None

# ===============================
# SIGNIFICANCE (same as before)
# ===============================
def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1) if n > 1 else 0.0
    tcrit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
    half = tcrit * s / np.sqrt(n) if n > 1 else np.nan
    return m, m - half, m + half, s, n

def paired_ci95(diff):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    m = np.mean(diff)
    s = np.std(diff, ddof=1) if n > 1 else 0.0
    tcrit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
    half = tcrit * s / np.sqrt(n) if n > 1 else np.nan
    return m, m - half, m + half, s, n

def cohens_d_paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    count = 0
    for ai in a:
        count += np.sum(ai > b) - np.sum(ai < b)
    return count / (len(a) * len(b) + 1e-12)

def fdr_bh(pvals, alpha=0.05):
    p = np.array(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    passed = ranked <= thresh
    crit = np.max(np.where(passed, np.arange(1, n+1), 0)) if np.any(passed) else 0
    cutoff = thresh[crit-1] if crit > 0 else 0
    qvals = np.empty_like(p)
    prev = 1.0
    for i in range(n-1, -1, -1):
        qvals[order[i]] = min(prev, ranked[i] * n / (i+1))
        prev = qvals[order[i]]
    return qvals, cutoff

def best_k_per_modality(results_df):
    agg = (results_df.groupby(['level','base_config','model','modality','k_requested'])
           .agg(mean_ba=('balanced_accuracy','mean'),
                n=('balanced_accuracy','count'))
           .reset_index())
    idx = agg.sort_values(['level','base_config','model','modality','mean_ba'],
                          ascending=[True,True,True,True,False]) \
             .groupby(['level','base_config','model','modality']).head(1)
    return idx[['level','base_config','model','modality','k_requested','mean_ba']]

def significance_comparisons(results_df, save_dir):
    best = best_k_per_modality(results_df)
    rows = []
    for (lvl, bc, mdl), sub in best.groupby(['level','base_config','model']):
        have = set(sub['modality'])
        def get_series(mod):
            r = sub[sub['modality']==mod]
            if r.empty: return None, None
            k = r['k_requested'].iloc[0]
            s = results_df[(results_df['level']==lvl) &
                           (results_df['base_config']==bc) &
                           (results_df['model']==mdl) &
                           (results_df['modality']==mod) &
                           (results_df['k_requested']==k)]
            s = s[['repeat','fold','balanced_accuracy']].dropna()
            return k, s
        pairs = [('motion','au'), ('all','motion')]
        if lvl == 'session':
            pairs.extend([('interpersonal','au'), ('interpersonal','motion')])

        for a_mod, b_mod in pairs:
            if a_mod in have and b_mod in have:
                kA, sA = get_series(a_mod)
                kB, sB = get_series(b_mod)
                if sA is None or sB is None: 
                    continue
                merged = pd.merge(sA, sB, on=['repeat','fold'], suffixes=('_A','_B'))
                if merged.empty: 
                    continue
                A = merged['balanced_accuracy_A'].values
                B = merged['balanced_accuracy_B'].values
                diff = A - B

                tstat, p_t = stats.ttest_rel(A, B, alternative='greater')
                try:
                    wstat, p_w = stats.wilcoxon(A, B, alternative='greater', zero_method='wilcox')
                except Exception:
                    p_w = np.nan

                d = cohens_d_paired(A, B)
                delta = cliffs_delta(A, B)
                md, lo, hi, sd, n = paired_ci95(diff)

                tA, pA = stats.ttest_1samp(A, 0.5, alternative='greater')
                tB, pB = stats.ttest_1samp(B, 0.5, alternative='greater')
                mA, loA, hiA, sdA, nA = mean_ci95(A)
                mB, loB, hiB, sdB, nB = mean_ci95(B)

                rows.append({
                    'level': lvl,
                    'base_config': bc,
                    'model': mdl,
                    'A_modality': a_mod, 'A_bestK': kA, 'A_meanBA': mA, 'A_CI95_low': loA, 'A_CI95_high': hiA, 'A_sd': sdA, 'A_n': nA, 'A_p_vs_chance': pA,
                    'B_modality': b_mod, 'B_bestK': kB, 'B_meanBA': mB, 'B_CI95_low': loB, 'B_CI95_high': hiB, 'B_sd': sdB, 'B_n': nB, 'B_p_vs_chance': pB,
                    'A_minus_B_mean': md, 'A_minus_B_CI95_low': lo, 'A_minus_B_CI95_high': hi, 'paired_sd': sd, 'paired_n': n,
                    'p_ttest_paired_greater': p_t, 'p_wilcoxon_greater': p_w,
                    'cohens_d_paired': d, 'cliffs_delta': delta
                })

    sig_df = pd.DataFrame(rows)
    if not sig_df.empty:
        qvals, _ = fdr_bh(sig_df['p_ttest_paired_greater'].fillna(1.0).values, alpha=0.05)
        sig_df['q_fdr_bh'] = qvals
        sig_df['significant_fdr_0.05'] = sig_df['q_fdr_bh'] <= 0.05

    out_path = os.path.join(save_dir, "significance_summary.csv")
    sig_df.to_csv(out_path, index=False)
    print(f"[significance] Saved {out_path}")
    return sig_df

# ===============================
# FEATURE USAGE SUMMARIES
# ===============================
def _summarize_feature_logs(save_dir):
    """
    Produces two CSVs:
      1) feature_selection_usage.csv
         - how often each engineered feature is selected across (repeat, fold),
           grouped by level/base_config/modality/k_requested
      2) top_features_overall.csv
         - same, but aggregated over k_requested to see global stability
    """
    if not FEATURE_LOGS:
        print("[feature-usage] No feature logs to summarize.")
        return None, None

    feat_df = pd.DataFrame(FEATURE_LOGS)
    run_df  = pd.DataFrame(RUN_LOGS).drop_duplicates()

    # Denominator: number of (repeat, fold) runs for each setting
    denom = (run_df
             .groupby(['level','base_config','modality','k_requested'])
             .size()
             .rename('n_runs')
             .reset_index())

    # Count selections
    sel_counts = (feat_df
                  .groupby(['level','base_config','modality','k_requested','feature'])
                  .size()
                  .rename('selected_count')
                  .reset_index())

    usage = pd.merge(sel_counts, denom,
                     on=['level','base_config','modality','k_requested'],
                     how='left')
    usage['selection_rate'] = usage['selected_count'] / usage['n_runs'].replace(0, np.nan)
    usage = usage.sort_values(['level','base_config','modality','k_requested','selection_rate'],
                              ascending=[True, True, True, True, False])

    usage.to_csv(os.path.join(save_dir, "feature_selection_usage.csv"), index=False)
    print(f"[feature-usage] Saved {os.path.join(save_dir, 'feature_selection_usage.csv')}")

    # Aggregate across k to see globally stable features per (level, base_config, modality)
    overall = (usage
               .groupby(['level','base_config','modality','feature'])
               .agg(total_selected=('selected_count','sum'),
                    total_runs=('n_runs','sum'))
               .reset_index())
    overall['global_selection_rate'] = overall['total_selected'] / overall['total_runs'].replace(0, np.nan)
    overall = overall.sort_values(['level','base_config','modality','global_selection_rate'],
                                  ascending=[True, True, True, False])
    overall.to_csv(os.path.join(save_dir, "top_features_overall.csv"), index=False)
    print(f"[feature-usage] Saved {os.path.join(save_dir, 'top_features_overall.csv')}")

    return usage, overall

# ===============================
# PLOTTING
# ===============================
def _normalize_k_for_plot(series):
    return pd.to_numeric(series, errors='coerce')

def _sig_star(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def plot_balacc_vs_k_by_modality(results_df, base_config, model_name, save_dir):
    df = results_df.copy()
    df = df[(df['base_config'] == base_config) & (df['model'] == model_name)]
    if df.empty:
        print(f"[plot] No data for base_config='{base_config}', model='{model_name}'")
        return
    df['k_numeric'] = _normalize_k_for_plot(df['k_requested'])
    for mod in df['modality'].unique():
        mask = (df['modality'] == mod) & (df['k_requested'].astype(str) == "all")
        if mask.any():
            k_eff_mean = df.loc[mask, 'k_effective'].mean()
            df.loc[mask, 'k_numeric'] = k_eff_mean
    g = df.groupby(['level','modality', 'k_requested','k_numeric'])['balanced_accuracy'].agg(['mean','std','count']).reset_index()
    g = g.sort_values('k_numeric')

    plt.figure(figsize=(8,5))
    for lvl in sorted(g['level'].unique()):
        for mod in ['au', 'motion', 'interpersonal', 'all']:
            sub = g[(g['level']==lvl) & (g['modality']==mod)]
            if sub.empty:
                continue
            x = sub['k_numeric'].values
            y = sub['mean'].values
            yerr = (sub['std'] / np.sqrt(sub['count'].clip(lower=1))).values
            plt.plot(x, y, marker='o', label=f"{lvl}-{mod}")
            plt.fill_between(x, (y - yerr), (y + yerr), alpha=0.15)

    plt.title(f"Balanced Accuracy vs Top-K — {base_config} / {model_name}")
    plt.xlabel("Top-K features (numeric; 'all'→effective)")
    plt.ylabel("Balanced Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Level–Modality", fontsize=8)
    fname = os.path.join(save_dir, f"ba_vs_k__{base_config}__{model_name}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[plot] Saved {fname}")

def plot_best_by_modality_bar(results_df, base_config, model_name, save_dir, sig_df=None):
    df = results_df.copy()
    df = df[(df['base_config'] == base_config) & (df['model'] == model_name)]
    if df.empty:
        print(f"[plot] No data for base_config='{base_config}', model='{model_name}'")
        return

    agg = df.groupby(['level','modality', 'k_requested']).agg(
        mean_ba=('balanced_accuracy','mean'),
        std_ba=('balanced_accuracy','std'),
        n=('balanced_accuracy','count')
    ).reset_index()

    best = agg.sort_values(['level','modality','mean_ba'], ascending=[True,True,False]) \
              .groupby(['level','modality']).head(1)

    plt.figure(figsize=(7,4))
    x_labels = [f"{lvl}-{mod}" for lvl, mod in best[['level','modality']].itertuples(index=False)]
    x = np.arange(len(best))
    y = best['mean_ba'].values
    se = (best['std_ba'] / np.sqrt(best['n'].clip(lower=1))).values
    plt.bar(x, y, yerr=se, alpha=0.9)
    plt.xticks(x, x_labels, rotation=20)
    plt.ylabel("Balanced Accuracy")
    plt.title(f"Best BA by Modality (per Level) — {base_config} / {model_name}")
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', alpha=0.2)

    if sig_df is not None and not sig_df.empty:
        sub = sig_df[(sig_df['base_config']==base_config) & (sig_df['model']==model_name)]
        pos = {lab:i for i,lab in enumerate(x_labels)}
        def y_top(i,j):
            return max(y[i]+se[i], y[j]+se[j]) + 0.03
        for lvl in best['level'].unique():
            to_draw = [('motion','au'), ('all','motion')]
            if lvl == 'session':
                to_draw.extend([('interpersonal','motion'), ('interpersonal','au')])
            for A,B in to_draw:
                row = sub[(sub['level']==lvl) & (sub['A_modality']==A) & (sub['B_modality']==B)]
                labelA, labelB = f"{lvl}-{A}", f"{lvl}-{B}"
                if not row.empty and (labelA in pos) and (labelB in pos):
                    i, j = pos[labelA], pos[labelB]
                    p = row['p_ttest_paired_greater'].values[0]
                    star = _sig_star(p)
                    y_max = y_top(i,j)
                    plt.plot([i, i, j, j], [y_max, y_max+0.01, y_max+0.01, y_max], lw=1)
                    plt.text((i+j)/2, y_max+0.015, star, ha='center', va='bottom')

    fname = os.path.join(save_dir, f"best_by_modality__{base_config}__{model_name}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[plot] Saved {fname}")

def make_all_plots(results_df, save_dir, sig_df=None):
    PLOT_MODELS = [
        "RandomForest_Conservative",
        "ExtraTrees",
        "SVM_RBF",
        "LogisticRegression_L2",
    ]
    base_configs = results_df['base_config'].unique().tolist()

    for bc in base_configs:
        for mdl in PLOT_MODELS:
            plot_balacc_vs_k_by_modality(results_df, bc, mdl, save_dir)
            plot_best_by_modality_bar(results_df, bc, mdl, save_dir, sig_df=sig_df)


def plot_feature_selection_counts(selection_counts_dict, output_path, top_n=50, figsize=(12, 8)):
    """
    Creates a horizontal bar plot of raw feature selection counts.
    """
    # Convert the input dict to a DataFrame
    df = pd.DataFrame({
        "raw_feature": list(selection_counts_dict.keys()),
        "selection_count": list(selection_counts_dict.values())
    })

    if "selection_count" not in df.columns:
        raise ValueError("Expected 'selection_count' column.")

    df = df[df["selection_count"] > 0]
    df = df.sort_values("selection_count", ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    def categorize_feature(feat: str):
        if "AU" in feat and feat.endswith("_r"):
            return "AU"
        elif any(x in feat.lower() for x in ['disp_', 'vel_', 'drift_', 'head_', 'shoulder_', 'hand_', 'x_', 'y_', 'z_']):
            return "Motion"
        elif any(x in feat.lower() for x in ['pair_', 'rel_', 'dist_', 'inter_', 'gap_', 'proximity']):
            return "Interpersonal"
        return "Other"

    df["category"] = df["raw_feature"].apply(categorize_feature)
    df["color"] = df["category"].map(CATEGORY_COLORS)

    row_height = 0.4  # You can tune this
    fig_height = max(6, len(df) * row_height)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(df["raw_feature"], df["selection_count"], color=df["color"], edgecolor='black')

    ax.set_xlabel("Selection Count", fontsize=14)
    ax.set_ylabel("Raw Feature", fontsize=14)
    ax.set_title("Top Feature Selection Frequency", fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # for bar in bars:
    #     width = bar.get_width()
    #     ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{int(width)}",
    #             va='center', ha='left', fontsize=11)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in CATEGORY_COLORS.items()]
    ax.legend(handles=patches, title="Feature Type", fontsize=11, title_fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✅ plot] Saved improved selection count plot to: {output_path}")
    
    
def plot_feature_weights(feature_weights_dict, output_path, top_n=50, figsize=(12, 8)):
    """
    Creates a horizontal bar plot of **sum of absolute weights** per raw feature.
    """
    def categorize_feature(feat: str):
        if "AU" in feat and feat.endswith("_r"):
            return "AU"
        elif any(x in feat.lower() for x in ['disp_', 'vel_', 'drift_', 'head_', 'shoulder_', 'hand_', 'x_', 'y_', 'z_']):
            return "Motion"
        elif any(x in feat.lower() for x in ['pair_', 'rel_', 'dist_', 'inter_', 'gap_', 'proximity']):
            return "Interpersonal"
        return "Other"

    # Convert to DataFrame
    data = []
    for feat, weights in feature_weights_dict.items():
        if isinstance(weights, (list, np.ndarray)):
            weight_sum = np.sum(np.abs(weights))
        else:
            weight_sum = abs(weights)
        if weight_sum > 0:
            data.append((feat, weight_sum))

    df = pd.DataFrame(data, columns=["raw_feature", "sum_abs_weight"])
    df = df.sort_values("sum_abs_weight", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    df["category"] = df["raw_feature"].apply(categorize_feature)
    df["color"] = df["category"].map(CATEGORY_COLORS)

    row_height = 0.4
    fig_height = max(6, len(df) * row_height)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    bars = ax.barh(df["raw_feature"], df["sum_abs_weight"], color=df["color"], edgecolor='black')

    ax.set_xlabel("Sum of Absolute Weights", fontsize=14)
    ax.set_ylabel("Raw Feature", fontsize=14)
    ax.set_title("Top Feature Importances (Logistic Regression)", fontsize=16, fontweight='bold', pad=2)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    ax.margins(y=0.01) 
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in CATEGORY_COLORS.items()]
    ax.legend(handles=patches, title="Feature Type", fontsize=11, title_fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✅ plot] Saved weight plot to: {output_path}")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    results_df, summary_df, model_summary = run_comprehensive_evaluation()

    if results_df is not None and not results_df.empty:
        sig_df = significance_comparisons(results_df, FOLDER_NAME)
        make_all_plots(results_df, FOLDER_NAME, sig_df=sig_df)

        print("\nDone. Files written to:")
        print(f"  - {os.path.join(FOLDER_NAME, 'comprehensive_results.csv')}")
        print(f"  - {os.path.join(FOLDER_NAME, 'config_summary.csv')}")
        print(f"  - {os.path.join(FOLDER_NAME, 'detailed_model_summary.csv')}")
        print(f"  - {os.path.join(FOLDER_NAME, 'significance_summary.csv')}")
        print(f"  - {os.path.join(FOLDER_NAME, 'feature_selection_usage.csv')}")
        print(f"  - {os.path.join(FOLDER_NAME, 'top_features_overall.csv')}")
        print(f"  - PNG figures in {FOLDER_NAME}/")
        
        # Improved raw feature analysis
        raw_csv_pattern = f"{FOLDER_NAME}/raw_feature_weights__*.csv"
        import glob
        raw_csv_files = glob.glob(raw_csv_pattern)
        
        if raw_csv_files:
            # Use the first available raw weights file
            raw_csv = raw_csv_files[0]
            
            # Create improved plots
            raw_plot = os.path.join(FOLDER_NAME, "raw_feature_contributions_improved.png")
            # plot_feature_selection_counts(csv_path=raw_csv, output_path=raw_plot, top_n=None)
            
            # Analyze by feature type
            feature_categories = analyze_raw_features_by_type(raw_csv, FOLDER_NAME)
            
            print(f"\n[✅ Raw Feature Analysis]")
            print(f"  - Motion features: {len(feature_categories['motion'])}")
            print(f"  - Facial features: {len(feature_categories['facial'])}")
            print(f"  - Interpersonal features: {len(feature_categories['interpersonal'])}")
            print(f"  - Other features: {len(feature_categories['other'])}")
        else:
            print("[⚠️] No raw feature weight CSV files found for analysis")