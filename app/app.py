import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Charger le données
df = pd.read_csv("Data/house_prices_kaggle.csv")
X_plot = df[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', 'GrLivArea', 'GarageCars']]
y = df['SalePrice']

# Charger le modèle
model = joblib.load("model.pkl")

st.title("Modèle de prédiction du prix des maisons")

st.write("📊 Entrez les caractéristiques du bien pour prédire son prix")

# Champs de saisie adaptés au dataset
lot_area = st.number_input("Surface du terrain", min_value=100, max_value=50000, step=50)
overall_qual = st.slider("Qualité globale (1 à 10)", min_value=1, max_value=10)
year_built = st.number_input("Année de construction", min_value=1800, max_value=2025, step=1)
total_rooms = st.number_input("Nombre total de pièces)", min_value=1, max_value=20, step=1)
gr_liv_area = st.number_input("Surface habitable (hors sous-sol)", min_value=20, max_value=500)
garage_cars = st.slider("Nombre de places de garage", min_value=0, max_value=5)

# Quand l'utilisateur clique sur le bouton
prediction_user = None

if st.button("Prédire le prix"):
    # mise en forme des données pour le modele
    X_user = np.array([[lot_area, overall_qual, year_built, total_rooms, gr_liv_area, garage_cars]])

    # Prédiction
    prediction_user = model.predict(X_user)[0]

    # Affichage
    st.success(f"Prix estimé : {prediction_user:,.0f} €")


# intigration du graphique dans Streamlit
# Nettoyage
df_clean = df[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', 'GrLivArea', 'GarageCars','SalePrice']].dropna()

X_plot = df_clean[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', 'GrLivArea', 'GarageCars']]
y_true = df_clean['SalePrice'].values
y_pred_all = model.predict(X_plot)

fig, ax = plt.subplots(figsize=(6,4))

# Points bleus : dataset
ax.scatter(y_true, y_pred_all, alpha=0.6, label="Maisons du dataset")

# Ligne parfaite
ax.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    linestyle="--",
    color="black"
)

# 🔴 Point rouge : maison utilisateur
if prediction_user is not None:
    ax.scatter(
        prediction_user,
        prediction_user,
        color="red",
        s=120,
        label="Votre maison"
    )

ax.set_xlabel("Prix réel")
ax.set_ylabel("Prix prédit")
ax.set_title("Prix réel vs prix prédit")
ax.legend()

st.pyplot(fig)