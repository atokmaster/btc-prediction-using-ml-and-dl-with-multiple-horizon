
# ======================== #
# 🔁 BTC AI Training 7D Model (Revised)
# ======================== #

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Dropout, Conv1D,
                                     MaxPooling1D, BatchNormalization, GlobalAveragePooling1D,
                                     LayerNormalization, MultiHeadAttention, Add)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# === Config
label_col = 'label_7d'
data_path = 'G:/BTC_AI_SIgnal/data/btc_1d_final.xlsx'
base_dir_ml = 'G:/BTC_AI_SIgnal/loop_results/Horizon_7D/from_1d/ML'
base_dir_dl = 'G:/BTC_AI_SIgnal/loop_results/Horizon_7D/from_1d/DL'
final_path = 'G:/BTC_AI_SIgnal/loop_results/Horizon_7D/final_comparison_7D.csv'

os.makedirs(base_dir_ml, exist_ok=True)
os.makedirs(base_dir_dl, exist_ok=True)

# === Load Data
df = pd.read_excel(data_path).dropna(subset=[label_col])
exclude_cols = [col for col in df.columns if col.startswith('label_') or col.startswith('return_') or col == 'datetime']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X_raw = df[feature_cols].values
y_raw = df[label_col].values

# ============ #
# 🔹 ML SECTION
# ============ #
X_scaled = StandardScaler().fit_transform(X_raw)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=42)

ml_models = {
    'xgboost': xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                                 use_label_encoder=False, scale_pos_weight=2),
    'lightgbm': LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'randomforest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

results = []

for name, model in ml_models.items():
    print(f"\n Training ML model: {name.upper()}")
    model_dir = os.path.join(base_dir_ml, name)
    os.makedirs(model_dir, exist_ok=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    model_path = os.path.join(model_dir, f"{name}_model_7d_from_1d.pkl")
    json.dump({'features': feature_cols, 'feature_len': len(feature_cols)}, open(os.path.join(model_dir, 'features.json'), 'w'), indent=2)
    pd.to_pickle(model, model_path)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title(f'Confusion Matrix – {name.upper()}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{name}_conf_matrix.png"))
    plt.close()

    results.append({
        'Model_Name': f"{name.upper()}_7D",
        'Model_Type': 'ML',
        'Model_Path': f"Horizon_7D/from_1d/ML/{name}/{name}_model_7d_from_1d.pkl",
        'Source': '1D',
        'Architecture': name.upper(),
        'Accuracy': round(acc, 4),
        'F1_Macro': round(f1, 4),
        'Feature_Len': len(feature_cols),
        'Epoch_Used': '-',
        'Sequence_Length': '-',
        'Dropout': '-'
    })

# ============ #
# 🔸 DL SECTION
# ============ #
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_lstm(seq_len, input_dim, dropout, units):
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        LSTM(units),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(3, activation='softmax')
    ])
    return model

def build_gru(seq_len, input_dim, dropout, units):
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        GRU(units),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(3, activation='softmax')
    ])
    return model

def build_cnn_lstm(seq_len, input_dim, dropout, units):
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(units),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(3, activation='softmax')
    ])
    return model

def build_transformer(seq_len, input_dim, dropout, units):
    inp = Input(shape=(seq_len, input_dim))
    x = Dense(units)(inp)
    x_attn = MultiHeadAttention(num_heads=4, key_dim=units // 2)(x, x)
    x = Add()([x, x_attn])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(units, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(3, activation='softmax')(x)
    return Model(inputs=inp, outputs=out)

model_dict = {
    'lstm': build_lstm,
    'gru': build_gru,
    'cnn_lstm': build_cnn_lstm,
    'transformer': build_transformer
}

sequence_len = 90
dropouts = [0.4, 0.5]
units_list = [16, 32, 64]
epochs = 150
batch_size = 32

X_seq, y_seq = create_sequences(X_scaled, y_raw, sequence_len)
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
input_dim = X_seq.shape[2]
class_weight = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

features_json = {
    "features": feature_cols,
    "feature_len": len(feature_cols)
}

for model_name, build_fn in model_dict.items():
    model_dir = os.path.join(base_dir_dl, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'features.json'), 'w') as f:
        json.dump(features_json, f, indent=2)

    for dr in dropouts:
        for units in units_list:
            fname = f"{model_name}{units}_model_ep{epochs}_7d_from_1d"
            model_path = os.path.join(model_dir, f"{fname}.keras")
            plot_path = os.path.join(model_dir, f"{fname}.png")

            model = build_fn(sequence_len, input_dim, dr, units)
            model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            es = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
            mc = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=epochs, batch_size=batch_size,
                                callbacks=[es, mc], class_weight=class_weight, verbose=0)

            used_epoch = len(history.history['val_accuracy'])
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            f1 = round(f1_score(y_val, y_pred, average='macro'), 4)

            # Save plot
            plt.figure()
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Val')
            plt.axvline(used_epoch-1, color='red', linestyle='--')
            plt.title(fname)
            plt.xlabel('Epoch'); plt.ylabel('Accuracy')
            plt.legend(); plt.tight_layout()
            plt.savefig(plot_path); plt.close()

            results.append({
                'Model_Name': f"{model_name.upper()}{units}_7D",
                'Model_Type': 'DL',
                'Model_Path': f"Horizon_7D/from_1d/DL/{model_name}/{fname}.keras",
                'Source': '1D',
                'Architecture': f"{model_name.upper()}_{units}",
                'Accuracy': round(history.history['val_accuracy'][-1], 4),
                'F1_Macro': f1,
                'Feature_Len': len(feature_cols),
                'Epoch_Used': used_epoch,
                'Sequence_Length': sequence_len,
                'Dropout': dr
            })

# === Save comparison file
df_result = pd.DataFrame(results)
if os.path.exists(final_path):
    df_old = pd.read_csv(final_path)
    df_result = pd.concat([df_old, df_result], ignore_index=True)
df_result.to_csv(final_path, index=False)
print(f" Final comparison 7D saved to: {final_path}")
