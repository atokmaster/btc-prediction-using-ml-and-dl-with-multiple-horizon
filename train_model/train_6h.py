import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# === Path Lokal
base_dir = r"G:\BTC_AI_SIgnal\loop_results\Horizon_6H\from_1h"
csv_path = r"G:\BTC_AI_SIgnal\loop_results\Horizon_6H\final_comparison_6H.csv"
data_path = r"G:\BTC_AI_SIgnal\data\btc_1h_final.xlsx"

# === Load Data
df = pd.read_excel(data_path)
target_col = 'label_6h'
excluded_cols = ['datetime', 'return_6h', 'label_6h', 'label_12h', 'return_12h']
feature_cols = [col for col in df.columns if col not in excluded_cols]
X = df[feature_cols].values
y = df[target_col].values.astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_cat = to_categorical(y, num_classes=3)
results = []

# === ML
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
models = {
    "xgboost": xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False),
    "lightgbm": lgb.LGBMClassifier(objective='multiclass', num_class=3),
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42)
}
for name, model in models.items():
    print(f"\n Training ML: {name.upper()}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    model_dir = os.path.join(base_dir, "ML", name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}_model_6h_from_1h.pkl")
    joblib.dump(model, model_path)

    with open(os.path.join(model_dir, "features.json"), 'w') as f:
        json.dump({"features": feature_cols, "feature_len": len(feature_cols)}, f)

    results.append({
        "Model_Name": f"{name.upper()}_6H",
        "Model_Type": "ML",
        "Accuracy": round(acc, 4),
        "F1_Macro": round(f1_macro, 4),
        "Source": "1H",
        "Architecture": name.upper(),
        "Model_Path": f"Horizon_6H/from_1h/ML/{name}/{name}_model_6h_from_1h.pkl",
        "Feature_Len": len(feature_cols),
        "Epoch_Used": "-",
        "Sequence_Length": "-",
        "Dropout": "-"
    })

# === DL
sequence_len = 60
dropout = 0.4
units_list = [16, 32, 64]
epochs = 150
patience = 7
batch_size = 32

architectures = {
    "lstm": LSTM,
    "bilstm": lambda units: Bidirectional(LSTM(units))
}
for arch_name, layer_fn in architectures.items():
    model_dir = os.path.join(base_dir, "DL", arch_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "features.json"), 'w') as f:
        json.dump({"features": feature_cols, "feature_len": len(feature_cols)}, f)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_len):
        X_seq.append(X_scaled[i:i+sequence_len])
        y_seq.append(y_cat[i+sequence_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, stratify=np.argmax(y_seq, axis=1), random_state=42)

    for units in units_list:
        print(f"\n Training DL: {arch_name.upper()} | Units={units}")
        model = Sequential([
            Input(shape=(sequence_len, X_seq.shape[2])),
            layer_fn(units),
            Dropout(dropout),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=0)
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

        used_epoch = len(history.history['val_accuracy'])
        y_pred = model.predict(X_test, verbose=0)
        y_true_cls = np.argmax(y_test, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true_cls, y_pred_cls)
        f1_macro = f1_score(y_true_cls, y_pred_cls, average='macro')

        filename = f"{arch_name}{units}_model_ep{used_epoch}_6h_from_1h.keras"
        filepath = os.path.join(model_dir, filename)
        model.save(filepath)

        results.append({
            "Model_Name": f"{arch_name.upper()}{units}_6H",
            "Model_Type": "DL",
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1_macro, 4),
            "Source": "1H",
            "Architecture": f"{arch_name.upper()}_{units}",
            "Model_Path": f"Horizon_6H/from_1h/DL/{arch_name}/{filename}",
            "Feature_Len": len(feature_cols),
            "Epoch_Used": used_epoch,
            "Sequence_Length": sequence_len,
            "Dropout": dropout
        })

# === Simpan hasil akhir
df_result = pd.DataFrame(results)
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    df_result = pd.concat([df_existing, df_result], ignore_index=True)
df_result.to_csv(csv_path, index=False)
print(f"\n Final comparison saved to: {csv_path}")
