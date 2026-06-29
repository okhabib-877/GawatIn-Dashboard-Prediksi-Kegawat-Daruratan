import os
import time
import urllib.parse
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Import Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Import Gemini
from google import genai

app = Flask(__name__)
CORS(app)

# =================================================================
# 1. SETUP MODEL MACHINE LEARNING (TRIASE)
# =================================================================
try:
    # INTEGRASI 1: Menyesuaikan dengan nama file model dari train_baru.py
    export_package = joblib.load('model_gawatin_selected.pkl')
    model = export_package['model']
    knn_imputer = export_package['imputer']
    scaler = export_package['scaler']
    
    # INTEGRASI 2: Mengambil daftar fitur secara dinamis dari train_baru.py
    fitur_awal_model = export_package['fitur_awal']
    fitur_final_model = export_package['fitur_final']
    print("✅ Model Ensemble (3 Naga) berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal load model ML: {e}")

# =================================================================
# 2. SETUP GEMINI AI & MEMORI CHAT (GAWATIN AI)
# =================================================================
os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6LyRIXATLCS2LCFKmf91X9apHwhN3PJrYM2W8VkC7KrmQ"
client = genai.Client()

tools = [{"google_search": {}}]

generation_config = {
    'temperature': 0.8,
    'max_output_tokens': 2048, 
    'top_p': 0.95,
}

Riwayat_Chat = []

# =================================================================
# 3. SETUP SELENIUM BROWSER GAIB (GAWATIN AI)
# =================================================================
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.page_load_strategy = 'eager'
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)

print("⏳ Memanaskan browser Selenium untuk GawatIn AI... Tunggu sebentar!")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
print("✅ Browser GawatIn AI siap!")

def scrape_alodokter(keyword):
    try:
        query = urllib.parse.quote(keyword)
        driver.get(f"https://www.alodokter.com/search?q={query}")
        time.sleep(1)
        
        artikel_pertama = driver.find_element(By.CLASS_NAME, 'search-result-title')
        link_artikel = artikel_pertama.get_attribute('href')
        
        driver.get(link_artikel)
        time.sleep(1)
        
        paragraf_elements = driver.find_elements(By.TAG_NAME, 'p')
        teks_hasil = " ".join([p.text for p in paragraf_elements[:5]])
        return teks_hasil
    except Exception as e:
        return "Data spesifik di Alodokter tidak ditemukan saat ini."


# =================================================================
# 4. ROUTING HALAMAN WEB UTAMA
# =================================================================
@app.route('/')
def home():
    return render_template('index.html')

# =================================================================
# 5. ENDPOINT API: TRIASE MACHINE LEARNING
# =================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        missing_fields = []
        
        # INTEGRASI 3: Hanya membentuk dictionary sesuai kebutuhan imputer dari train_baru.py
        patient_data = {col: np.nan for col in fitur_awal_model}

        if data.get('age') in [None, ""]: missing_fields.append('age')
        else: patient_data['AGE'] = float(data.get('age'))

        if data.get('temperature') in [None, ""]: missing_fields.append('temperature')
        else: patient_data['TEMPF'] = (float(data.get('temperature')) * 9/5) + 32
            
        if data.get('heartrate') in [None, ""]: missing_fields.append('heartrate')
        else: patient_data['PULSE'] = float(data.get('heartrate'))
            
        if data.get('resprate') in [None, ""]: missing_fields.append('resprate')
        else: patient_data['RESPR'] = float(data.get('resprate'))
            
        if data.get('o2sat') in [None, ""]: missing_fields.append('o2sat')
        else: patient_data['POPCT'] = float(data.get('o2sat'))
            
        if data.get('sbp') in [None, ""]: missing_fields.append('sbp')
        else: patient_data['BPSYS'] = float(data.get('sbp'))
            
        if data.get('dbp') in [None, ""]: missing_fields.append('dbp')
        else: patient_data['BPDIAS'] = float(data.get('dbp'))
            
        # Catatan: Fitur 'pain' tetap ditampung logikanya agar UI 'Skenario Terburuk' 
        # di index.html tetap jalan dengan mulus, meski ML tidak memakainya.
        if data.get('pain') in [None, ""]: missing_fields.append('pain')

        # Proses Data Sesuai Pipeline train_baru.py
        df_input = pd.DataFrame([patient_data], columns=fitur_awal_model)
        df_scaled = scaler.transform(df_input)
        df_imputed = knn_imputer.transform(df_scaled)
        df_clean = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=fitur_awal_model)

        df_clean['TEMPC'] = (df_clean['TEMPF'] - 32) * 5.0 / 9.0
        df_clean['shock_index'] = np.where(df_clean['BPSYS'] > 0, df_clean['PULSE'] / df_clean['BPSYS'], 0)
        df_clean['MAP'] = (df_clean['BPSYS'] + 2 * df_clean['BPDIAS']) / 3

        # INTEGRASI 4: Memfilter fitur berdasarkan 'fitur_final_model' elit dari train_baru.py
        X_final = df_clean[fitur_final_model]

        pred_raw = int(model.predict(X_final)[0])
        probabilities = model.predict_proba(X_final)[0]
        confidence_percentage = float(round(max(probabilities) * 100, 1))

        hypotheticals = []
        is_incomplete = len(missing_fields) > 0
        pred_acuity = pred_raw + 1 

        all_probabilities = {
            "Acuity 1": float(round(probabilities[0] * 100, 1)),
            "Acuity 2": float(round(probabilities[1] * 100, 1)),
            "Acuity 3": float(round(probabilities[2] * 100, 1)),
            "Acuity 4": float(round(probabilities[3] * 100, 1)),
            "Acuity 5": float(round(probabilities[4] * 100, 1))
        }

        # INTEGRASI 5: Menyesuaikan simulasi skenario ketika data tidak lengkap
        if is_incomplete:
            scenarios_data = [
                {
                    'id': 'buruk',
                    'vals': {'age': 65.0, 'temperature': 39.5, 'heartrate': 120.0, 'resprate': 28.0, 'o2sat': 92.0, 'sbp': 85.0, 'dbp': 50.0, 'pain': 8.0},
                    'desc': {'age': 'pasien lansia', 'temperature': 'demam tinggi', 'heartrate': 'jantung berdebar', 'resprate': 'napas terengah', 'o2sat': 'napas berat', 'sbp': 'tensi anjlok', 'dbp': 'keringat dingin', 'pain': 'nyeri hebat'}
                },
                {
                    'id': 'sehat',
                    'vals': {'age': 25.0, 'temperature': 36.5, 'heartrate': 75.0, 'resprate': 16.0, 'o2sat': 99.0, 'sbp': 120.0, 'dbp': 80.0, 'pain': 0.0},
                    'desc': {'age': 'usia muda', 'temperature': 'suhu normal', 'heartrate': 'jantung normal', 'resprate': 'napas teratur', 'o2sat': 'pernapasan lega', 'sbp': 'tensi normal', 'dbp': 'tidak lemas', 'pain': 'tanpa nyeri'}
                }
            ]

            acuity_titles = { 1: 'Kondisi Kritis', 2: 'Risiko Tinggi', 3: 'Kondisi Mendesak', 4: 'Kondisi Stabil', 5: 'Tidak Gawat' }

            for sc_data in scenarios_data:
                hypo_dict = df_clean.iloc[0].to_dict()
                symptom_texts = []

                for field in missing_fields:
                    val = sc_data['vals'][field]
                    if field == 'age': hypo_dict['AGE'] = val
                    elif field == 'temperature': hypo_dict['TEMPF'] = (val * 9/5) + 32
                    elif field == 'heartrate': hypo_dict['PULSE'] = val
                    elif field == 'resprate': hypo_dict['RESPR'] = val
                    elif field == 'o2sat': hypo_dict['POPCT'] = val
                    elif field == 'sbp': hypo_dict['BPSYS'] = val
                    elif field == 'dbp': hypo_dict['BPDIAS'] = val
                    symptom_texts.append(sc_data['desc'][field])

                h_tempc = (hypo_dict['TEMPF'] - 32) * 5.0 / 9.0
                h_shock = hypo_dict['PULSE'] / hypo_dict['BPSYS'] if hypo_dict['BPSYS'] > 0 else 0
                h_map = (hypo_dict['BPSYS'] + 2 * hypo_dict['BPDIAS']) / 3

                # Rekonstruksi struktur data untuk prediksi hipotesis
                h_features_dict = {
                    'AGE': hypo_dict.get('AGE', 0),
                    'TEMPF': hypo_dict.get('TEMPF', 0),
                    'PULSE': hypo_dict.get('PULSE', 0),
                    'RESPR': hypo_dict.get('RESPR', 0),
                    'BPSYS': hypo_dict.get('BPSYS', 0),
                    'BPDIAS': hypo_dict.get('BPDIAS', 0),
                    'POPCT': hypo_dict.get('POPCT', 0),
                    'TEMPC': h_tempc,
                    'shock_index': h_shock,
                    'MAP': h_map
                }
                
                h_features_df = pd.DataFrame([h_features_dict])
                h_features_selected = h_features_df[fitur_final_model]

                h_pred_raw = int(model.predict(h_features_selected)[0])
                h_acuity = h_pred_raw + 1
                h_prob = round(max(model.predict_proba(h_features_selected)[0]) * 100, 1)

                if len(symptom_texts) > 1:
                    symptoms_str = ", ".join(symptom_texts[:-1]) + ", dan " + symptom_texts[-1]
                elif len(symptom_texts) == 1:
                    symptoms_str = symptom_texts[0]
                else:
                    symptoms_str = "kondisi spesifik"

                dynamic_title = acuity_titles.get(h_acuity, 'Status Kondisi')
                hypotheticals.append({
                    'title': dynamic_title,
                    'symptoms': symptoms_str,
                    'acuity': h_acuity,
                    'confidence': h_prob
                })

            hypotheticals.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'acuity': pred_acuity,
            'confidence': confidence_percentage,
            'all_probabilities': all_probabilities,
            'is_incomplete': is_incomplete,
            'hypotheticals': hypotheticals
        })

    except Exception as e:
        print(f"❌ Error Triase: {str(e)}")
        return jsonify({'error': str(e)}), 500

# =================================================================
# 6. ENDPOINT API: GAWATIN CHATBOT
# =================================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    global Riwayat_Chat
    data = request.json
    pesan_user = data.get('message', '')
    
    if not pesan_user:
        return jsonify({"status": "error", "message": "Pesan kosong"}), 400
    
    Riwayat_Chat.append(f"Pasien: {pesan_user}")
    
    data_medis_mentah = scrape_alodokter(pesan_user)
    konteks_percakapan = "\n".join(Riwayat_Chat)
    
    prompt_untuk_ai = f"""
    Kamu adalah GawatIn AI, chatbot kesehatan yang bertindak seperti teman dekat atau tempat cerita yang sangat hangat, ramah, penuh empati, dan menenangkan. 
    Pasien mungkin akan mengeluh atau curhat tentang kondisinya, maka dengarkan dengan tulus.
    
    Berikut riwayat obrolan kalian:
    {konteks_percakapan}
    
    Data referensi medis untuk keluhan terbaru: 
    "{data_medis_mentah}"
    
    GAYA MENJAWAB GAWATIN AI (PENTING):
    1. Berikan respon awal yang penuh kehangatan dan rasa peduli. Buat pasien merasa didengarkan curhatnya.
    2. JANGAN BERTELE-TELE. Jika dari riwayat obrolan kamu sudah mengetahui 1 atau 2 gejala utama, LANGSUNG berikan edukasi, saran praktis, atau solusi konkret yang selinear dengan data Alodokter/Google Search.
    3. Jika gejalanya masih abu-abu, tanya 1 detail penting dengan sopan, tapi langsung susun dengan perkiraan solusinya.
    4. WAJIB memisahkan responmu menjadi beberapa bagian dengan tanda [SPLIT] agar tampil sebagai berbubble-bubble chat yang mengalir santai.
    5. Di akhir solusi, tetap ingatkan dengan lembut untuk periksa ke dokter jika kondisi memburuk.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt_untuk_ai,
            config=genai.types.GenerateContentConfig(
                tools=tools,
                temperature=generation_config['temperature'],
                top_p=generation_config['top_p']
            )
        )
        jawaban_mentah = response.text
        Riwayat_Chat.append(f"GawatIn AI: {jawaban_mentah}")
        
        daftar_bubble = [b.strip() for b in jawaban_mentah.split('[SPLIT]') if b.strip()]
        
    except Exception as e:
        if "503" in str(e):
            daftar_bubble = ["Duh, maaf banget ya... [SPLIT] Otak GawatIn AI mendadak nge-blank karena antrean pasien lagi padat. [SPLIT] Boleh tolong kirim ulang ceritamu yang tadi? Terima kasih ya."]
        else:
            daftar_bubble = [f"Aduh, sepertinya sistemku sedikit lelah: {e}"]
    
    return jsonify({
        "status": "success",
        "reply": daftar_bubble
    })

if __name__ == '__main__':
    # HANYA SATU PORT SEKARANG (5000)
    app.run(host='0.0.0.0', port=5000, debug=True)