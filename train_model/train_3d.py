# === Train 3D Horizon from 4H and 1D datasets (ML + DL) ===

import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# === Config ===
label_col = 'label_3d'
seq_len = 60
dropout = 0.4
epochs = 150
patience = 7
batch_size = 32
units_list = [16, 32, 64]
base_dir = "G:/BTC_AI_SIgnal/loop_results/Horizon_3D"
os.makedirs(base_dir, exist_ok=True)

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(seq_length, len(X)):
        Xs.append(X[i-seq_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def save_features_json(path, features, length):
    with open(os.path.join(path, "features.json"), 'w') as f:
        json.dump({"features": features, "feature_len": length}, f)

def train_ml_model(data_path, source_tag):
    df = pd.read_excel(data_path).dropna(subset=[label_col])
    feature_cols = [col for col in df.columns if col not in ['datetime'] and not col.startswith('return_') and not col.startswith('label_')]
    X = df[feature_cols].values
    y = df[label_col].values
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'xgboost': xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False),
        'lightgbm': LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'randomforest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    result = []
    for name, model in models.items():
        print(f" Training {name.upper()} ({source_tag})")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        model_dir = os.path.join(base_dir, f"from_{source_tag}/ML/{name}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{name}_model_3d_from_{source_tag}.pkl")
        joblib.dump(model, model_path)
        save_features_json(model_dir, feature_cols, len(feature_cols))
        result.append({
            "Model_Name": f"{name.upper()}_3D",
            "Model_Type": "ML",
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1, 4),
            "Source": source_tag.upper(),
            "Architecture": name.upper(),
            "Model_Path": f"Horizon_3D/from_{source_tag}/ML/{name}/{name}_model_3d_from_{source_tag}.pkl",
            "Feature_Len": len(feature_cols),
            "Epoch_Used": "-", "Sequence_Length": "-", "Dropout": "-"
        })
    return result

def train_dl_models(data_path, source_tag):
    df = pd.read_excel(data_path).dropna(subset=[label_col])
    feature_cols = [col for col in df.columns if col not in ['datetime'] and not col.startswith('return_') and not col.startswith('label_')]
    X = df[feature_cols].values
    y = df[label_col].values.astype(int)
    X_scaled = StandardScaler().fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    input_shape = X_train.shape[1:]
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y), y=y)))

    def build_lstm(units): return Sequential([
        Input(shape=input_shape), LSTM(units), Dropout(dropout),
        Dense(units, activation='relu'), Dropout(dropout), Dense(3, activation='softmax')
    ])
    def build_gru(units): return Sequential([
        Input(shape=input_shape), GRU(units), Dropout(dropout),
        Dense(units, activation='relu'), Dropout(dropout), Dense(3, activation='softmax')
    ])
    def build_cnn_lstm(units): return Sequential([
        Input(shape=input_shape), Conv1D(64, 3, activation='relu'), MaxPooling1D(2),
        LSTM(units), Dropout(dropout), Dense(units, activation='relu'),
        Dropout(dropout), Dense(3, activation='softmax')
    ])
    def build_transformer(units):
        inp = Input(shape=input_shape)
        x = Dense(units)(inp)
        attn = MultiHeadAttention(num_heads=4, key_dim=units//2)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)
        out = Dense(3, activation='softmax')(x)
        return Model(inputs=inp, outputs=out)

    builders = {
        'lstm': build_lstm, 'gru': build_gru,
        'cnn_lstm': build_cnn_lstm, 'transformer': build_transformer
    }

    result = []
    for name, build_fn in builders.items():
        model_dir = os.path.join(base_dir, f"from_{source_tag}/DL/{name}")
        os.makedirs(model_dir, exist_ok=True)
        save_features_json(model_dir, feature_cols, len(feature_cols))
        for units in units_list:
            model = build_fn(units)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            es = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
            history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat),
                                epochs=epochs, batch_size=batch_size,
                                callbacks=[es], class_weight=class_weights, verbose=0)
            used_epoch = len(history.history['val_accuracy'])
            val_acc = round(history.history['val_accuracy'][-1], 4)
            y_pred = model.predict(X_test, verbose=0)
            f1_macro = round(f1_score(np.argmax(y_test_cat, axis=1), np.argmax(y_pred, axis=1), average='macro'), 4)
            fname = f"{name}{units}_model_ep{used_epoch}_3d_from_{source_tag}.keras"
            model.save(os.path.join(model_dir, fname))
            result.append({
                "Model_Name": f"{name.upper()}{units}_3D",
                "Model_Type": "DL",
                "Accuracy": val_acc,
                "F1_Macro": f1_macro,
                "Source": source_tag.upper(),
                "Architecture": f"{name.upper()}_{units}",
                "Model_Path": f"Horizon_3D/from_{source_tag}/DL/{name}/{fname}",
                "Feature_Len": len(feature_cols),
                "Epoch_Used": used_epoch,
                "Sequence_Length": seq_len,
                "Dropout": dropout
            })
    return result

# === RUN ALL ===
results = []
results += train_ml_model("G:/BTC_AI_SIgnal/data/btc_4h_final.xlsx", "4h")
results += train_dl_models("G:/BTC_AI_SIgnal/data/btc_4h_final.xlsx", "4h")
results += train_ml_model("G:/BTC_AI_SIgnal/data/btc_1d_final.xlsx", "1d")
results += train_dl_models("G:/BTC_AI_SIgnal/data/btc_1d_final.xlsx", "1d")

# === Save final_comparison_3D.csv ===
df_result = pd.DataFrame(results)
csv_path = os.path.join(base_dir, "final_comparison_3D.csv")
df_result.to_csv(csv_path, index=False)
print(f" All results saved to: {csv_path}")
