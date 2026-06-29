import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. BACA & FILTER DATASET
df = pd.read_csv('DATA_SAMPLING_300_PER_LEVEL.csv')
df = df[df['IMMEDR'].isin([1, 2, 3, 4, 5])].copy()
y = df['IMMEDR'] - 1 # Zero-based indexing

# KUNCI STABILITAS 1: Reset index agar data tidak melompat setelah difilter
df.reset_index(drop=True, inplace=True)

# 2. SELEKSI FITUR & PREPROCESSING
fitur_tersedia = ['AGE', 'TEMPF', 'PULSE', 'RESPR', 'BPSYS', 'BPDIAS', 'POPCT']
X = df[fitur_tersedia].copy()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_scaled)
X_clean = pd.DataFrame(scaler.inverse_transform(X_imputed), columns=fitur_tersedia)

# 3. FEATURE ENGINEERING
X_clean['TEMPC'] = (X_clean['TEMPF'] - 32) * 5.0 / 9.0
X_clean['shock_index'] = np.where(X_clean['BPSYS'] > 0, X_clean['PULSE'] / X_clean['BPSYS'], 0)
X_clean['MAP'] = (X_clean['BPSYS'] + 2 * X_clean['BPDIAS']) / 3

features_raw = ['AGE', 'TEMPC', 'PULSE', 'RESPR', 'BPSYS', 'BPDIAS', 'POPCT', 'shock_index', 'MAP']
X_final_raw = X_clean[features_raw]

# =================================================================
# TAHAP 1: FEATURE IMPORTANCE YANG TRANSPARAN
# =================================================================
print("="*60)
print("🔍 TAHAP 1: MENCARI FEATURE IMPORTANCE (SELEKSI FITUR)")
print("="*60)

# KUNCI STABILITAS 2: random_state=42 pada Juri Penilai
rf_tester = RandomForestClassifier(random_state=42, n_estimators=100)
rf_tester.fit(X_final_raw, y)

# Urutkan fitur dari yang paling penting ke yang paling rendah
feature_importances = pd.DataFrame({
    'Fitur': features_raw,
    'Skor_Penting': rf_tester.feature_importances_
}).sort_values(by='Skor_Penting', ascending=False)

print("\n📊 Peringkat Pengaruh Fitur (Dari Tertinggi ke Terendah):")
print(feature_importances.to_string(index=False))

# --- MEMISAHKAN FITUR YANG DIPAKAI DAN DIBUANG ---
fitur_terbaik = feature_importances['Fitur'].head(len(features_raw) - 2).tolist()
fitur_dibuang = feature_importances['Fitur'].tail(2).tolist()

print("\n" + "-"*60)
print(f"✅ FITUR ELIT YANG DIPAKAI : {fitur_terbaik}")
print(f"🗑️ FITUR YANG DIBUANG      : {fitur_dibuang}")
print("-"*60)

X_final_selected = X_clean[fitur_terbaik]

# =================================================================
# TAHAP 2: PEMBELAJARAN 3 NAGA
# =================================================================
# KUNCI STABILITAS 3: random_state=42 pada Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final_selected, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🤖 TAHAP 2: MELATIH ENSEMBLE (3 NAGA)")

# KUNCI STABILITAS 4: Seluruh model dasar dikunci dengan random_state=42
lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

ensemble_model = VotingClassifier(
    estimators=[('lgb', lgb_model), ('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

# =================================================================
# TAHAP 3: EVALUASI & EXPORT
# =================================================================
print("\n" + "="*60)
print(f"🎯 AKURASI KONSISTEN & STABIL: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("="*60)
print("\nRincian Classification Report:\n", classification_report(y_test, y_pred))

# BUNGKUS DAN SIMPAN
export_package = {
    'model': ensemble_model,
    'imputer': imputer,
    'scaler': scaler,
    'fitur_awal': fitur_tersedia, 
    'fitur_final': fitur_terbaik # Simpan list fitur elitnya
}
joblib.dump(export_package, 'model_gawatin_selected.pkl')
print("\n✅ BUNGKUSAN SELESAI! Model stabil disimpan sebagai 'model_gawatin_selected.pkl'")