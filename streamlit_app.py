# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed, parallel_backend
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import time
from deep_translator import GoogleTranslator
import os
from typing import Dict, Any

# Traducteurs
en_to_fr = GoogleTranslator(target='fr')
fr_to_en = GoogleTranslator(target='en')

# Configuration de la page
st.set_page_config(
    page_title="🌱 BOOT AGRIC - Agriculture de Précision",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- (Ton CSS inchangé) ---
st.markdown("""
<style>
    /* Import de Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    :root {
        --primary-green: #2E7D32;
        --secondary-green: #4CAF50;
        --accent-gold: #FFC107;
        --bg-gradient: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        --card-shadow: 0 8px 32px rgba(46, 125, 50, 0.1);
    }
    .main .block-container {
        background: var(--bg-gradient);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--card-shadow);
    }
    .main-header {
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(46, 125, 50, 0.3);
    }
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(46, 125, 50, 0.15);
    }
    .metric-container {
        background: linear-gradient(135deg, #F8F9FA, #FFFFFF);
        border-left: 5px solid var(--secondary-green);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #E8F5E8, #F1F8E9);
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.6s ease-out; }
    .status-excellent { color: #2E7D32; font-weight: bold; }
    .status-good { color: #4CAF50; font-weight: bold; }
    .status-warning { color: #FF9800; font-weight: bold; }
    .status-poor { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header fade-in">
    <h1>🌱 BOOT AGRIC</h1>
    <p>🚀 Intelligence Artificielle pour l'Agriculture de Précision</p>
</div>
""", unsafe_allow_html=True)
# --- end CSS ---

# URLs des modèles Azure
MODEL_URLS = {
    'recommendation': 'https://wenzedevstockage.blob.core.windows.net/keys/crop_recommadation_model.pkl',
    'yield': 'https://wenzedevstockage.blob.core.windows.net/keys/rendement_model_optimized.pkl',
    'nutrients': 'https://wenzedevstockage.blob.core.windows.net/keys/estimation_ressources_model_optimized.pkl'
}
DATASET_URL = "https://wenzedevstockage.blob.core.windows.net/keys/FinalcropPrediction.csv"
DATASET_CLIMATE = "https://wenzedevstockage.blob.core.windows.net/keys/data_rendement.csv"

# ---- Utilitaires / sécurité ----
DEFAULT_REQUEST_TIMEOUT = 15  # secondes
MAX_THREADS = min(4, max(1, (os.cpu_count() or 2)))  # cap raisonnable

# Utiliser st.cache_resource pour garder les modèles en mémoire entre les runs
@st.cache_resource
def download_and_load_model(url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT):
    """Télécharge un pkl depuis une URL (avec timeout) et le charge via joblib."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        # joblib.load à partir de bytes
        obj = joblib.load(io.BytesIO(resp.content))
        return obj
    except Exception as e:
        # Ne raise pas pour ne pas casser l'app - on renvoie None
        st.warning(f"⚠️ Erreur téléchargement/modèle depuis {url}: {e}")
        return None

def safe_requests_csv(url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.error(f"❌ Erreur chargement CSV {url}: {e}")
        return pd.DataFrame()

# Chargement des modèles en parallèle (threading backend pour compatibilité)
def load_all_models_parallel(model_urls: Dict[str, str]) -> Dict[str, Any]:
    keys = list(model_urls.keys())
    urls = [model_urls[k] for k in keys]

    # joblib Parallel en mode threads (safe pour Streamlit)
    with parallel_backend('threading', n_jobs=MAX_THREADS):
        results = Parallel(n_jobs=len(urls), prefer='threads')(
            delayed(download_and_load_model)(u) for u in urls
        )
    # mappage clé->modèle
    return {k: m for k, m in zip(keys, results)}

# Fonctions de prédiction (identiques à ton code, avec petits ajustements)
def predict_crop_recommendation(sample, model_data):
    """Prédiction de recommandation de culture"""
    try:
        if model_data is None:
            return None
        sample_copy = sample.copy()
        # Certains de tes modèles attendent d'autres clés - robustesse
        if 'weather' in sample_copy:
            del sample_copy['weather']
        model = model_data.get('model', None) if isinstance(model_data, dict) else model_data.get('model', None)
        preprocessor = model_data.get('preprocessor', None)
        label_encoder = model_data.get('label_encoder', None)
        top_indices = model_data.get('top_indices', None)
        input_features = model_data.get('input_features', None)

        if model is None or preprocessor is None or label_encoder is None or input_features is None:
            st.warning("⚠️ Structure du modèle de recommandation inattendue.")
            return None

        missing_features = [f for f in input_features if f not in sample_copy]
        if missing_features:
            st.warning(f"⚠️ Features manquantes: {missing_features}")
            return None

        sample_df = pd.DataFrame([sample_copy])[input_features]
        sample_prep = preprocessor.transform(sample_df)
        sample_selected = sample_prep[:, top_indices]

        pred_class = model.predict(sample_selected)[0]
        pred_proba = model.predict_proba(sample_selected)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        proba_dict = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(pred_proba)}

        return {'class': pred_label, 'probabilities': proba_dict}
    except Exception as e:
        st.error(f"❌ Erreur prédiction recommandation: {e}")
        return None

def predict_yield(sample, to_plant, model_data):
    """Prédiction du rendement (même logique que ton code)"""
    try:
        if model_data is None:
            return None, None
        models = model_data.get("models", {})
        preprocessor = model_data.get("preprocessor", None)
        weights = model_data.get("weights", [1]*len(models))
        input_features = model_data.get("input_features", [])
        # Harmonisation
        sample_input = pd.DataFrame([{
            'Temperature': sample['temperature'],
            'Rainfall': sample['rainfall'],
            'Humidity': sample['humidity'],
            'Soil': sample['soil'],
            'Weather': sample['weather'],
            'Crop': to_plant
        }])
        rename_map = {
            "Temperature": "Temperature (°C)",
            "Rainfall": "Rainfall (mm)",
            "Humidity": "Humidity (%)",
            "Soil": "Soil Type",
            "Weather": "Weather Condition",
            "Crop": "Crop Type"
        }
        sample_input.rename(columns=rename_map, inplace=True)

        def create_domain_expert_features(df):
            df = df.copy()
            temp = df['Temperature (°C)']
            rain = df['Rainfall (mm)']
            humidity = df['Humidity (%)']
            df['GDD'] = np.maximum(0, temp - 10)
            df['GDD_Accumulated'] = df['GDD'] * 30
            df['Water_Stress_Index'] = np.maximum(0, (temp * 2 - rain)) / 100
            df['Moisture_Adequacy'] = np.minimum(1, rain / (temp * 10 + 50))
            df['VPD'] = 0.611 * np.exp(17.27 * temp / (temp + 237.3)) * (1 - humidity/100)
            df['WUE'] = (rain * humidity) / (temp * 100 + 1)
            df['Plant_Comfort'] = np.exp(-0.5 * ((temp - 22) / 8) ** 2)
            df['Temp_Rain_Efficiency'] = temp * rain / 1000
            df['Climate_Synergy'] = (temp / 30) * (rain / 200) * (humidity / 80)
            df['Temp_Too_Low'] = (temp < 15).astype(float)
            df['Temp_Optimal'] = ((temp >= 20) & (temp <= 30)).astype(float)
            df['Temp_Too_High'] = (temp > 35).astype(float)
            df['Rain_Drought'] = (rain < 100).astype(float)
            df['Rain_Optimal'] = ((rain >= 150) & (rain <= 300)).astype(float)
            df['Rain_Excess'] = (rain > 400).astype(float)
            df['Humidity_Low'] = (humidity < 50).astype(float)
            df['Humidity_Optimal'] = ((humidity >= 60) & (humidity <= 80)).astype(float)
            df['Humidity_High'] = (humidity > 85).astype(float)
            df['Perfect_Conditions'] = df['Temp_Optimal'] * df['Rain_Optimal'] * df['Humidity_Optimal']
            df['Stress_Conditions'] = df['Temp_Too_High'] + df['Rain_Drought'] + df['Humidity_Low']
            base_yield = 3 + 2 * df['Climate_Synergy']
            ph_by_crop = {'Wheat': 6.5, 'Rice': 6.0, 'Corn': 6.8, 'Soybean': 6.3}
            df['Soil_pH_Optimal'] = df['Crop Type'].map(ph_by_crop).fillna(6.5)
            df['pH_Deviation'] = abs(6.5 - df['Soil_pH_Optimal'])
            df['Available_N'] = 40 + 8 * base_yield + 0.1 * rain
            df['Available_P'] = 25 + 3 * base_yield + 0.05 * rain
            df['Available_K'] = 60 + 5 * base_yield + 0.08 * rain
            df['NPK_Balance'] = df['Available_N'] / (df['Available_P'] + 1) / (df['Available_K'] + 1)
            return df

        sample_input = create_domain_expert_features(sample_input)
        for col in input_features:
            if col not in sample_input.columns:
                sample_input[col] = 0
        sample_input = sample_input[input_features]
        if preprocessor is not None:
            X_processed = preprocessor.transform(sample_input)
        else:
            X_processed = sample_input.values

        preds = [m.predict(X_processed) for m in models.values()] if models else [np.zeros((1,))]
        y_pred_final = np.average(preds, axis=0, weights=weights)
        y_val = float(y_pred_final[0])
        if y_val < 3:
            yield_class = "faible"
        elif y_val < 7:
            yield_class = "moyen"
        else:
            yield_class = "élevé"
        return y_val, yield_class
    except Exception as e:
        st.error(f"❌ Erreur prédiction rendement: {e}")
        return None, None

def predict_nutrients_streamlit(sample, model_data):
    """Prédiction des besoins en nutriments"""
    try:
        if model_data is None:
            return None
        rf_model = model_data.get('rf_model', None)
        gb_model = model_data.get('gb_model', None)
        et_model = model_data.get('et_model', None)
        weights = model_data.get('weights', [1,1,1])
        scaler_X = model_data.get('scaler_X', None)
        scaler_y = model_data.get('scaler_y', None)
        le_crop = model_data.get('label_encoder', None)
        n_crops = model_data.get('n_crops', 0)
        target_names = model_data.get('target_names', [])

        sample_copy = sample.copy()
        if le_crop is None:
            st.warning("⚠️ label_encoder absent dans model_data nutriments")
            return None
        if sample_copy['Crops'] not in le_crop.classes_:
            st.warning(f"❌ Culture '{sample_copy['Crops']}' non reconnue.")
            return None

        sample_copy['Crops'] = le_crop.transform([sample_copy['Crops']])[0]
        sample_df = pd.DataFrame([sample_copy])[['Temperature', 'Humidity', 'pH', 'Rainfall(cm)', 'Crops']]

        sample_enhanced = sample_df.copy()
        sample_enhanced['Water_Stress'] = (sample_df['Temperature'] - 20) / (sample_df['Humidity'] / 100 + 1e-8)
        sample_enhanced['pH_Optimal'] = np.abs(sample_df['pH'] - 6.5)
        sample_enhanced['Temp_Optimal'] = np.abs(sample_df['Temperature'] - 25)
        sample_enhanced['Drought_Index'] = sample_df['Temperature'] / (sample_df['Rainfall(cm)'] + 1e-8)
        sample_enhanced['N_Uptake_Factor'] = sample_df['Temperature'] * sample_df['Humidity'] * (sample_df['pH'] - 5)
        sample_enhanced['P_Availability'] = sample_df['pH'] * sample_df['Rainfall(cm)'] / sample_df['Temperature']
        sample_enhanced['K_Mobility'] = sample_df['Humidity'] * sample_df['Rainfall(cm)'] / (sample_df['Temperature'] + 1)
        sample_enhanced['Cu_Solubility'] = sample_df['pH'] * sample_df['Humidity']
        sample_enhanced['Fe_Availability'] = (7 - sample_df['pH']) * sample_df['Rainfall(cm)']
        sample_enhanced['Mg_Leaching'] = sample_df['Rainfall(cm)'] / (sample_df['pH'] + 1)
        sample_enhanced['Temp_Squared'] = sample_df['Temperature'] ** 2
        sample_enhanced['pH_Squared'] = sample_df['pH'] ** 2
        sample_enhanced['Humidity_Squared'] = sample_df['Humidity'] ** 2

        for crop_id in range(n_crops):
            sample_enhanced[f'Crop_{crop_id}'] = (sample_df['Crops'] == crop_id).astype(int)

        if scaler_X is not None:
            sample_scaled = scaler_X.transform(sample_enhanced)
        else:
            sample_scaled = sample_enhanced.values

        rf_pred = rf_model.predict(sample_scaled) if rf_model is not None else np.zeros((1, len(target_names)))
        gb_pred = gb_model.predict(sample_scaled) if gb_model is not None else np.zeros((1, len(target_names)))
        et_pred = et_model.predict(sample_scaled) if et_model is not None else np.zeros((1, len(target_names)))

        final_pred = (weights[0] * rf_pred + weights[1] * gb_pred + weights[2] * et_pred)
        if scaler_y is not None:
            pred_scaled = scaler_y.inverse_transform(final_pred)
        else:
            pred_scaled = final_pred
        pred_nutrients = np.expm1(np.maximum(pred_scaled[0], 0))
        return dict(zip(target_names, pred_nutrients))
    except Exception as e:
        st.error(f"❌ Erreur prédiction nutriments: {e}")
        return None

# --- Sidebar inputs (comme ton code) ---
st.sidebar.markdown("## 🌡️ Paramètres Environnementaux")
with st.sidebar:
    st.markdown("### 🌤️ Conditions Climatiques")
    df = safe_requests_csv(DATASET_URL)
    df_climate = safe_requests_csv(DATASET_CLIMATE)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("🌡️ Température (°C)", min_value=5, max_value=45, value=26, step=1)
        humidity = st.slider("💧 Humidité (%)", min_value=30, max_value=95, value=66, step=1)
    with col2:
        ph = st.slider("⚗️ pH du sol", min_value=4.0, max_value=9.0, value=6.8, step=0.1)
        rainfall = st.slider("🌧️ Précipitations (mm/an)", min_value=200, max_value=3000, value=1500, step=50)

    st.markdown("### 🌍 Conditions du Sol")
    soil_type = st.selectbox("🏔️ Type de sol", list(df.soil.unique()) if not df.empty else ["loam", "sandy", "clay"])
    weather = st.selectbox("☀️ Conditions météo", list(df_climate['Weather Condition'].unique()) if not df_climate.empty else ["Sunny", "Cloudy", "Rain"])

    st.markdown("### 🧪 Nutriments Actuels (NPK)")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Azote (N)", min_value=0, max_value=300, value=137)
    with col2:
        P = st.number_input("Phosphore (P)", min_value=0, max_value=150, value=85)
    with col3:
        K = st.number_input("Potassium (K)", min_value=0, max_value=400, value=175)

# Préparation des données
sample_data = {
    'N': N,
    'P': P,
    'K': K,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall,
    'soil': fr_to_en.translate(soil_type),
    'weather': fr_to_en.translate(weather)
}

# calcul
def calcul_economic(hectare, crop_type: None):

    prix_par_kilo_rdc_usd = {
        'rice': 1.00,  # Riz local et importé.
        'maize': 0.70,  # Maïs, prix souvent bas car c'est une culture de base.
        'chickpea': 2.00,  # Pois chiche, prix élevé car moins courant.
        'kidneybeans': 2.60, # Haricots rouges.
        'pigeonpeas': 2.00, # Pois d'Angole.
        'mothbeans': .70, # Données non disponibles.
        'mungbean': 2.50, # Haricot mungo.
        'blackgram': 5.50, # Un type de haricot noir de Goma, prix élevé.
        'lentil': 4.00, # Lentilles, souvent importées.
        'pomegranate': 5.00, # Grenade, rare et chère.
        'banana': 1.00, # Banane plantain, varie selon la taille et le marché.
        'mango': 1.50, # Mangue.
        'grapes': 8.00, # Raisins, importés et très chers.
        'watermelon': 1.00, # Pastèque, prix souvent au fruit entier.
        'muskmelon': 4.00, # Melon, rare et cher.
        'apple': 3.00, # Pommes, importées.
        'orange': 1.50, # Oranges, prix à la pièce.
        'papaya': 1.50, # Papaye, prix à la pièce.
        'coconut': 1.00, # Noix de coco, prix à la pièce.
        'cotton': 1.50, # Coton, prix de la fibre brute, non vendu au détail.
        'jute': .50, # Jute, prix de la fibre, non vendu au détail.
        'coffee': 2.50  # Café Arabica pour le producteur, prix de la fève verte.
    }
    une_tonne = 1000
    prix_par_kilos = prix_par_kilo_rdc_usd[str(crop_type).lower()]
    # return hectare * une_tonne * prix_par_kilos
    return (hectare - (hectare * 5/100)) * une_tonne * prix_par_kilos

# Onglets (identique)
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommandation", "📈 Rendement", "🧪 Nutriments", "📊 Dashboard"])

# --- TAB 1 ---
with tab1:
    st.markdown("## 🎯 Recommandation de Culture Optimale")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔮 Analyser & Recommander Complet", key="recommend_btn"):
            with st.spinner("🧠 Chargement des modèles et analyse..."):
                # 1) charger tous les modèles EN PARALLÈLE (threading)
                models = load_all_models_parallel(MODEL_URLS)

                recommendation_model = models.get('recommendation')
                yield_model = models.get('yield')
                nutrients_model = models.get('nutrients')

                if recommendation_model and yield_model and nutrients_model:
                    # Exécuter la recommandation (nécessaire pour yield & nutrients)
                    rec = predict_crop_recommendation(sample_data, recommendation_model)
                    if not rec:
                        st.error("❌ Échec de la recommandation.")
                    else:
                        recommended_crop = rec['class']
                        st.success(f"🌾 **Culture recommandée: {en_to_fr.translate(recommended_crop.upper())}**")

                        fig = px.bar(
                            x=[ en_to_fr.translate(i) for i in list(rec['probabilities'].keys())],
                            y=list(rec['probabilities'].values()),
                            title="🎯 Probabilités par Culture",
                            color=list(rec['probabilities'].values()),
                            color_continuous_scale="Greens"
                        )
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig, use_container_width=True)

                        st.session_state.recommended_crop = recommended_crop

                        # 2) Calculs lourds (rendement + nutriments) en parallèle (threads)
                        with parallel_backend('threading', n_jobs=2):
                            results = Parallel(n_jobs=2, prefer='threads')(
                                delayed(lambda fn, *a, **kw: fn(*a, **kw)) (f, *args)
                                for f, args in [
                                    (predict_yield, (sample_data, recommended_crop, yield_model)),
                                    (predict_nutrients_streamlit, ({'Temperature': temperature,
                                                                     'Humidity': humidity,
                                                                     'pH': ph,
                                                                     'Rainfall(cm)': rainfall / 10,
                                                                     'Crops': recommended_crop}, nutrients_model))
                                ]
                            )
                        # results[0] = (predicted_yield, yield_class)
                        predicted_yield, yield_class = results[0] if results and results[0] is not None else (None, None)
                        predicted_nutrients = results[1] if len(results) > 1 else None

                        # Affichage rendement
                        if predicted_yield is not None:
                            col_y1, col_y2, col_y3 = st.columns(3)
                            with col_y1:
                                st.metric("🏆 Rendement", f"{predicted_yield:.2f} t/ha")
                            with col_y2:
                                class_colors = {"faible": "🔴", "moyen": "🟡", "élevé": "🟢"}
                                st.metric("📊 Classe", f"{class_colors.get(yield_class, '⚪')} {yield_class.upper()}")
                            with col_y3:
                                economic_value = calcul_economic(predicted_yield, recommended_crop)
                                st.metric("💰 Valeur", f"{economic_value:.0f} USD/ha")
                        else:
                            st.warning("⚠️ Rendement non disponible.")

                        # Affichage nutriments
                        if predicted_nutrients:
                            col_n1, col_n2, col_n3 = st.columns(3)
                            with col_n1:
                                st.metric("🟦 Azote (N)", f"{predicted_nutrients.get('Nitrogen(N)',0):.1f} kg/ha")
                            with col_n2:
                                st.metric("🟧 Phosphore (P)", f"{predicted_nutrients.get('phosphorus (P)',0):.1f} kg/ha")
                            with col_n3:
                                st.metric("🟪 Potassium (K)", f"{predicted_nutrients.get('Potassium(K)',0):.1f} kg/ha")

                            nutrients_df = pd.DataFrame([
                                {'Nutriment': en_to_fr.translate(k.split('(')[0]), 'Besoin': v, 'Type': 'Macro' if k in ['Nitrogen(N)', 'phosphorus (P)', 'Potassium(K)'] else 'Micro'}
                                for k, v in predicted_nutrients.items()
                            ])
                            fig_nutrients = px.bar(nutrients_df, x='Nutriment', y='Besoin', color='Type',
                                                  title="🧪 Profil Nutritionnel Complet",
                                                  color_discrete_map={'Macro': '#4CAF50', 'Micro': '#2196F3'})
                            fig_nutrients.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                            st.plotly_chart(fig_nutrients, use_container_width=True)
                        else:
                            st.warning("⚠️ Nutriments non disponibles.")
                else:
                    st.error("❌ Impossible de charger tous les modèles depuis Azure.")
    with col2:
        st.markdown("""
        <div class="prediction-card" style="background: rgba(240, 248, 255, 0.9); border-left: 4px solid #4CAF50;">
            <h3 style="color: #2E7D32;">🧠 Comment ça marche ?</h3>
            <p style="color: #1565C0;">Notre IA analyse :</p>
            <ul style="color: #37474F;">
                <li>🌡️ Conditions climatiques</li>
                <li>🌍 Type de sol</li>
                <li>🧪 Nutriments disponibles</li>
                <li>📊 Données historiques</li>
            </ul>
            <p style="color: #2E7D32;"><strong>Recommandation en temps réel !</strong></p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: Rendement (unchanged logic, uses cached model load when needed) ---
with tab2:
    st.markdown("## 📈 Prédiction du Rendement Agricole")
    if 'recommended_crop' in st.session_state:
        to_plant = st.session_state.recommended_crop
        st.info(f"🌾 Culture sélectionnée: **{en_to_fr.translate(to_plant)}** (de la recommandation)")
    else:
        to_plant = st.selectbox("🌾 Choisir une culture", ["maize", "wheat", "rice", "barley", "cotton"])

    col1, col2, col3 = st.columns(3)
    if st.button("📊 Prédire le Rendement", key="yield_btn"):
        with st.spinner("🔄 Calcul du rendement potentiel..."):
            # Charger seulement le modèle rendement (cache_resource fait le reste)
            yield_model = download_and_load_model(MODEL_URLS['yield'])
            if yield_model:
                predicted_yield, yield_class = predict_yield(sample_data, to_plant, yield_model)
                with col1:
                    st.metric(label="🏆 Rendement Prédit", value=f"{predicted_yield:.2f} t/ha", delta=f"Culture: {en_to_fr.translate(to_plant)}")
                with col2:
                    class_colors = {"faible": "🔴", "moyen": "🟡", "élevé": "🟢"}
                    st.metric(label="📊 Classification", value=f"{class_colors.get(yield_class, '⚪')} {yield_class.upper()}")
                with col3:
                    
                    economic_value = calcul_economic(predicted_yield, to_plant)
                    st.metric(label="💰 Valeur Estimée", value=f"{economic_value:.0f} USD/ha")
                # Graphique...
                categories = ['Température', 'Humidité', 'Précipitations', 'pH', 'Nutriments']
                scores = [
                    min(100, max(0, 100 - abs(temperature - 25) * 3)),
                    min(100, max(0, humidity)),
                    min(100, max(0, rainfall / 20)),
                    min(100, max(0, 100 - abs(ph - 6.5) * 15)),
                    min(100, max(0, (N + P + K) / 6))
                ]
                fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="🎯 Score des Conditions Optimales", font_family="Poppins")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Modèle rendement indisponible.")

# --- TAB 3: Nutriments ---
with tab3:
    st.markdown("## 🧪 Estimation des Besoins en Nutriments")
    if st.button("🔬 Analyser les Besoins", key="nutrients_btn"):
        with st.spinner("🧪 Analyse des besoins nutritionnels..."):
            nutrients_model = download_and_load_model(MODEL_URLS['nutrients'])
            if nutrients_model:
                crop_for_nutrients = st.session_state.get('recommended_crop', 'maize')
                sample_for_nutrients = {'Temperature': temperature, 'Humidity': humidity, 'pH': ph, 'Rainfall(cm)': rainfall / 10, 'Crops': crop_for_nutrients}
                predicted_nutrients = predict_nutrients_streamlit(sample_for_nutrients, nutrients_model)
                if predicted_nutrients:
                    st.success(f"🌾 Analyse pour: **{crop_for_nutrients}**")
                    st.markdown("### 🔴 Macronutriments (NPK)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🟦 Azote (N)", f"{predicted_nutrients['Nitrogen(N)']:.1f}", "kg/ha")
                    with col2:
                        st.metric("🟧 Phosphore (P)", f"{predicted_nutrients['phosphorus (P)']:.1f}", "kg/ha")
                    with col3:
                        st.metric("🟪 Potassium (K)", f"{predicted_nutrients['Potassium(K)']:.1f}", "kg/ha")
                    nutrients_df = pd.DataFrame([{'Nutriment': en_to_fr.translate(k.split('(')[0]), 'Besoin': v, 'Type': 'Macro' if k in ['Nitrogen(N)', 'phosphorus (P)', 'Potassium(K)'] else 'Micro'} for k, v in predicted_nutrients.items()])
                    fig = px.bar(nutrients_df, x='Nutriment', y='Besoin', color='Type', title="🧪 Profil Nutritionnel Recommandé", color_discrete_map={'Macro': '#4CAF50', 'Micro': '#2196F3'})
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Prédiction nutriments échouée.")
            else:
                st.error("❌ Modèle nutriments indisponible.")

# --- TAB 4: Dashboard (conserve logique) ---
with tab4:
    st.markdown("## 📊 Dashboard Agriculture de Précision")
    if st.button("🚀 Analyse Complète", key="full_analysis"):
        with st.spinner("🔄 Analyse complète en cours..."):
            # Charger en parallèle (rapide)
            models = load_all_models_parallel(MODEL_URLS)
            recommendation_model = models.get('recommendation')
            yield_model = models.get('yield')
            nutrients_model = models.get('nutrients')
            if recommendation_model and yield_model and nutrients_model:
                rec = predict_crop_recommendation(sample_data, recommendation_model)
                if rec:
                    recommended_crop = rec['class']
                    yield_pred, yield_class = predict_yield(sample_data, recommended_crop, yield_model)
                    sample_nutrients = {'Temperature': temperature, 'Humidity': humidity, 'pH': ph, 'Rainfall(cm)': rainfall / 10, 'Crops': recommended_crop}
                    nutrients = predict_nutrients_streamlit(sample_nutrients, nutrients_model)
                else:
                    st.error("❌ Recommandation échouée.")
                    recommended_crop = "unknown"
                    yield_pred, yield_class = (0, "inconnu")
                    nutrients = None
            else:
                st.error("❌ Chargement modèles échoué.")
                recommended_crop = "unknown"
                yield_pred, yield_class = (0, "inconnu")
                nutrients = None

            st.markdown("### 🎯 Résumé Exécutif")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌾 Culture Optimale", en_to_fr.translate(recommended_crop.upper()))
            with col2:
                st.metric("📈 Rendement", f"{yield_pred:.1f} t/ha", f"Classe: {yield_class}")
            with col3:
                st.metric("💰 Valeur/ha", f"{calcul_economic(yield_pred, recommended_crop):.0f} USD")
            with col4:
                risk_score = max(0, min(100, 85 - abs(temperature - 25) * 2 - abs(ph - 6.5) * 10))
                st.metric("⚡ Score Risque", f"{risk_score:.0f}/100")

            # Graphiques...
            col1, col2 = st.columns(2)
            with col1:
                conditions = ['Température', 'Humidité', 'pH', 'Précipitations']
                scores = [min(100, max(0, 100 - abs(temperature - 25) * 3)),
                          humidity,
                          min(100, max(0, 100 - abs(ph - 6.5) * 15)),
                          min(100, max(0, rainfall / 30))]
                fig1 = go.Figure(data=go.Scatterpolar(r=scores, theta=conditions, fill='toself'))
                fig1.update_layout(title="🌡️ Conditions Environnementales", polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                npk_data = pd.DataFrame({
                    'Nutriment': ['Azote', 'Phosphore', 'Potassium'],
                    'Actuel': [N, P, K],
                    'Recommandé': [
                        nutrients['Nitrogen(N)'] if nutrients else N * 1.1,
                        nutrients['phosphorus (P)'] if nutrients else P * 1.1,
                        nutrients['Potassium(K)'] if nutrients else K * 1.1
                    ]
                })
                fig2 = px.bar(npk_data, x='Nutriment', y=['Actuel', 'Recommandé'], title="🧪 Balance NPK", barmode='group', color_discrete_sequence=['#FF7043', '#4CAF50'])
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 💡 Recommandations Actionables")
            recommendations = []
            if temperature > 35:
                recommendations.append("🌡️ **Température élevée** : Considérez l'irrigation par aspersion pour refroidir")
            if ph < 6.0:
                recommendations.append("⚗️ **Sol acide** : Apportez de la chaux pour augmenter le pH")
            if humidity < 50:
                recommendations.append("💧 **Faible humidité** : Augmentez l'irrigation ou utilisez du paillis")
            if rainfall < 600:
                recommendations.append("🌧️ **Précipitations faibles** : Système d'irrigation obligatoire")
            if not recommendations:
                recommendations.append("✅ **Conditions optimales** : Parfait pour la culture !")
            for rec in recommendations:
                st.markdown(f"- {rec}")

# Footer (inchangé)
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🎯 Précision du Modèle")
    st.markdown("""
    - **Recommandation**: 94.2% accuracy
    - **Rendement**: R² = 0.89
    - **Nutriments**: MSE < 0.05
    """)
with col2:
    st.markdown("### 🌍 Cultures Supportées")
    st.markdown("""
    - Céréales (blé, maïs, riz, orge)
    - Légumineuses (soja, haricots)
    - Cultures industrielles (coton)
    - Et 15+ autres...
    """)
with col3:
    st.markdown("### 🚀 Technologie")
    st.markdown("""
    - **IA**: Ensemble de modèles ML
    - **Cloud**: Azure Storage
    - **Temps réel**: < 2 secondes
    - **Fiabilité**: 99.9% uptime
    """)
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, #E8F5E8, #F1F8E9); border-radius: 10px;">
    <p style="color: #2E7D32; font-weight: bold;">🌱 BOOT AGRIC - Révolutionnez votre agriculture avec l'IA 🚀</p>
</div>
""", unsafe_allow_html=True)
