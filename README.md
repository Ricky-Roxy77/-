Lucky-Jet
import streamlit as st
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Configuration de la page ---
st.set_page_config(
    page_title="✈️ LUCKY JET",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Style personnalisé ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(to right, #00FF00, #00FFFF);
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #00CC00, #00CCFF);
    }
    h1, h2, h3 {
        color: #00FF00;
        text-shadow: 0 0 5px #00FF00;
    }
    .stSlider>div>div>div>div {
        background-color: #333;
    }
    .stAlert {
        background-color: #1E1E1E;
        border-left: 4px solid #00FF00;
    }
    .multiplier-display {
        font-size: 48px;
        font-weight: bold;
        color: #00FF00;
        text-shadow: 0 0 10px #00FF00;
        text-align: center;
        padding: 20px;
        background: rgba(0, 255, 0, 0.1);
        border-radius: 15px;
        border: 2px solid #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# --- Fonctions pour LUCKY JET ---
def simuler_crash(multiplicateur_max=100.0):
    """Simule un multiplicateur de crash aléatoire."""
    return random.uniform(1.00, multiplicateur_max)

def generer_donnees_entrainement(nb_echantillons=10000, multiplicateur_max=100.0):
    """Génère des données pour entraîner le modèle IA."""
    données = []
    for _ in range(nb_echantillons):
        # Génère une séquence de 5 multiplicateurs précédents
        sequence = [random.uniform(1.00, multiplicateur_max) for _ in range(5)]
        # Multiplicateur actuel (à prédire)
        multiplicateur_actuel = random.uniform(1.00, multiplicateur_max)
        # Caractéristiques : moyenne et écart-type des 5 précédents
        moyenne = np.mean(sequence)
        ecart_type = np.std(sequence)
        dernier = sequence[-1]
        données.append({
            "moyenne_précédente": moyenne,
            "écart_précédent": ecart_type,
            "dernier_multiplicateur": dernier,
            "multiplicateur_actuel": multiplicateur_actuel
        })
    return pd.DataFrame(données)

def entraîner_modèle(nb_echantillons=10000):
    """Entraîne un modèle Random Forest pour prédire le multiplicateur."""
    données = generer_donnees_entrainement(nb_echantillons)
    X = données[["moyenne_précédente", "écart_précédent", "dernier_multiplicateur"]]
    y = données["multiplicateur_actuel"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modèle = RandomForestRegressor(n_estimators=100, random_state=42)
    modèle.fit(X_train, y_train)
    score = modèle.score(X_test, y_test)
    return modèle, score

def prédire_prochain_multiplicateur(modèle, historique):
    """Prédit le prochain multiplicateur de crash."""
    if len(historique) < 5:
        return random.uniform(1.00, 100.0)
    moyenne = np.mean(historique[-5:])
    ecart_type = np.std(historique[-5:])
    dernier = historique[-1]
    prédiction = modèle.predict([[moyenne, ecart_type, dernier]])[0]
    return max(1.00, prédiction)  # Le multiplicateur ne peut pas être < 1.00

# --- Interface Streamlit ---
st.title("✈️ LUCKY JET")
st.markdown("""
---
**Bienvenue dans LUCKY JET !** 🎉
Parie sur le multiplicateur avant que l'avion ne **CRASH** ! Encaisse tes gains avant qu'il ne soit trop tard.
""")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    multiplicateur_max = st.slider("Multiplicateur max", 10.0, 500.0, 100.0, 10.0)
    st.markdown("---")
    st.markdown("### 💡 Conseils")
    st.info("""
    **Stratégie de base :**
    - Encaisser tôt (1.20x-1.50x) → **Faible risque, faible gain**.
    - Encaisser tard (5.00x-10.00x) → **Haut risque, haut gain**.
    - Utilise l'IA pour optimiser tes décisions !
    """)

# --- Onglet 1 : Jeu en Direct ---
tab1, tab2, tab3 = st.tabs(["✈️ Jeu en Direct", "🤖 Prédiction IA", "📊 Statistiques"])

with tab1:
    st.markdown("### 🎮 Jeu en Direct")
    st.markdown("""
    **Règles :**
    - L'avion décolle et son multiplicateur augmente.
    - **Encaisse (CASH OUT)** avant qu'il ne crash pour gagner !
    - Si tu ne cash pas à temps, tu perds ta mise.
    """)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if "multiplicateur" not in st.session_state:
            st.session_state.multiplicateur = 1.00
            st.session_state.historique = []
            st.session_state.capital = 10000
            st.session_state.mise = 100
            st.session_state.en_cours = False
            st.session_state.gain = 0

        if st.session_state.en_cours:
            st.markdown(f'<div class="multiplier-display">{st.session_state.multiplicateur:.2f}x</div>', unsafe_allow_html=True)
            st.progress(st.session_state.multiplicateur / multiplicateur_max)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💰 CASH OUT", key="cash_out"):
                    st.session_state.gain = st.session_state.mise * st.session_state.multiplicateur
                    st.session_state.capital += st.session_state.gain
                    st.session_state.historique.append(st.session_state.multiplicateur)
                    st.session_state.en_cours = False
                    st.success(f"✅ Gain : +{st.session_state.gain:.0f} FCFA ! Nouveau solde : {st.session_state.capital:.0f} FCFA")
            with col_b:
                if st.button("❌ ANNULER"):
                    st.session_state.capital -= st.session_state.mise
                    st.session_state.historique.append(1.00)  # Crash immédiat
                    st.session_state.en_cours = False
                    st.error(f"❌ Perte : -{st.session_state.mise:.0f} FCFA ! Nouveau solde : {st.session_state.capital:.0f} FCFA")

        else:
            st.markdown(f'<div class="multiplier-display">Prêt à jouer !</div>', unsafe_allow_html=True)
            mise = st.number_input("Mise (FCFA)", min_value=10, value=100, step=10, key="mise_input")
            if st.button("🚀 DÉMARRER LE JEU", key="start_game"):
                st.session_state.mise = mise
                st.session_state.multiplicateur = 1.00
                st.session_state.en_cours = True
                st.session_state.gain = 0
                st.rerun()

            if st.session_state.historique:
                st.markdown("### 📜 Derniers multiplicateurs")
                st.line_chart(pd.DataFrame({"Multiplicateur": st.session_state.historique[-10:]}).T)

    with col1:
        st.metric("Solde", f"{st.session_state.capital:.0f} FCFA")
    with col3:
        st.metric("Dernier gain", f"{st.session_state.gain:.0f} FCFA")

    # Simulation automatique du multiplicateur
    if st.session_state.en_cours:
        time.sleep(0.5)
        st.session_state.multiplicateur += random.uniform(0.01, 0.1)
        if st.session_state.multiplicateur >= multiplicateur_max:
            st.session_state.multiplicateur = multiplicateur_max
            st.session_state.capital -= st.session_state.mise
            st.session_state.historique.append(multiplicateur_max)
            st.session_state.en_cours = False
            st.error(f"💥 CRASH à {multiplicateur_max:.2f}x ! Perte : -{st.session_state.mise:.0f} FCFA")
        st.rerun()

# --- Onglet 2 : Prédiction IA ---
with tab2:
    st.markdown("### 🤖 Prédiction IA")
    st.markdown("""
    **Comment ça marche ?**
    L'IA analyse les **derniers multiplicateurs** pour prédire le prochain crash.
    Elle te conseille quand **encaisser (CASH OUT)** pour maximiser tes gains.
    """)

    if "modèle" not in st.session_state:
        st.info("⚠️ L'IA n'est pas encore entraînée. Clique sur le bouton ci-dessous.")
        if st.button("🧠 Entraîner l'IA"):
            with st.spinner("Entraînement de l'IA en cours..."):
                modèle, score = entraîner_modèle(nb_echantillons=10000)
                st.session_state.modèle = modèle
                st.session_state.score_ia = score
            st.success(f"✅ IA entraînée avec un score de **{score:.2%}** !")

    if "modèle" in st.session_state:
        st.markdown(f"**Score de l'IA : {st.session_state.score_ia:.2%}**")

        if st.session_state.historique and len(st.session_state.historique) >= 5:
            prédiction = prédire_prochain_multiplicateur(st.session_state.modèle, st.session_state.historique)
            st.markdown(f"**Prédiction du prochain crash : {prédiction:.2f}x**")

            # Conseils basés sur la prédiction
            if prédiction < 1.5:
                st.warning("⚠️ **Conseil** : Le crash devrait arriver tôt. **Encaisse maintenant !**")
            elif prédiction < 3.0:
                st.info("ℹ️ **Conseil** : Risque modéré. Tu peux attendre un peu.")
            else:
                st.success("✅ **Conseil** : Le crash devrait arriver tard. **Attends pour un gros gain !**")

            # Seuil d'encaisse recommandé
            seuil = prédiction * 0.8
            st.markdown(f"**Seuil d'encaisse recommandé : {seuil:.2f}x**")
        else:
            st.info("⚠️ **Attends d'avoir au moins 5 multiplicateurs dans l'historique** pour une prédiction précise.")

# --- Onglet 3 : Statistiques ---
with tab3:
    st.markdown("### 📊 Statistiques")
    if st.session_state.historique:
        df = pd.DataFrame({"Multiplicateur": st.session_state.historique})
        st.markdown("#### 📈 Évolution des multiplicateurs")
        st.line_chart(df)

        st.markdown("#### 📉 Distribution des multiplicateurs")
        fig, ax = plt.subplots()
        ax.hist(st.session_state.historique, bins=20, color='#00FF00', alpha=0.7, edgecolor='white')
        ax.set_xlabel("Multiplicateur")
        ax.set_ylabel("Fréquence")
        ax.set_facecolor('#1E1E1E')
        ax.grid(True, alpha=0.3, color='white')
        st.pyplot(fig)

        st.markdown("#### 💰 Historique des gains")
        gains = []
        capital_initial = 10000
        capital = capital_initial
        for mult in st.session_state.historique:
            if mult == 1.00:  # Perte
                capital -= 100
                gains.append(-100)
            else:
                gain = 100 * mult
                capital += gain
                gains.append(gain)
        st.line_chart(pd.DataFrame({"Gain": gains, "Capital": [capital_initial + sum(gains[:i+1]) for i in range(len(gains))]}))

        st.markdown(f"**Capital initial : {capital_initial} FCFA**")
        st.markdown(f"**Capital actuel : {capital:.0f} FCFA**")
        st.markdown(f"**Gain total : {capital - capital_initial:.0f} FCFA**")
    else:
        st.info("⚠️ Aucune donnée disponible. Joue quelques parties d'abord !")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>✈️ <strong>LUCKY JET</strong> - Application développée pour Frédéric Kouamé</p>
    <p>⚠️ Cette application est à but éducatif. Les résultats ne garantissent pas des gains réels.</p>
</div>
""", unsafe_allow_html=True)
