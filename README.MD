## Rapport du projet 



# Prédiction du prix des maisons – PriceVision

## Objectif du projet
PriceVision est une application de data science permettant de prédire le prix d’une maison à partir de
ses caractéristiques principales (surface du terrain, qualité générale, année de construction, nombre de
pièces, etc.).
Ce projet a été réalisé dans un objectif professionnel, afin de :
mettre en pratique la machine learning avec Python
construire un projet data de bout en bout
déployer un modèle de prédiction via une application web interactive (Streamlit)
## Approche et méthodologie
Le projet suit un workflow classique en data science :
1. Exploration des données (EDA)
2. Sélection et préparation des features
3. Entraînement de plusieurs modèles
4. Comparaison des performances
5. Choix du meilleur modèle
6. Déploiement dans une application Streamlit
## Données utilisées
Le dataset contient des informations sur des biens immobiliers, notamment :
LotArea : surface du terrain
OverallQual : qualité générale du bien (1 à 10)YearBuilt : année de construction
TotRmsAbvGrd : nombre total de pièces
SalePrice : prix de vente (variable cible)
Les variables catégorielles ont été volontairement exclues afin de débuter avec un modèle simple et
interprétable.
## Modèles testés
Deux modèles de régression ont été entraînés et comparés :
#### Linear Regression
Modèle simple et interprétable
Très bonnes performances sur ce dataset
#### Random Forest Regressor
Modèle plus complexe
Moins performant ici en raison de la taille réduite du dataset
Modèle retenu
Après comparaison à l’aide des métriques RMSE et R², le modèle Linear Regression a été retenu car il
offre :
- de meilleures performances
- une meilleure stabilité
- une interprétation plus claire des résultats
- Évaluation du modèle

Les performances ont été évaluées à l’aide des métriques suivantes :
MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² (coefficient de détermination)

Une visualisation Prix réel vs Prix prédit permet de juger visuellement la qualité du modèle.
## Application Streamlit
Une application web interactive a été développée avec Streamlit.
#### Fonctionnalités :
- Saisie des caractéristiques d’une maison
- Prédiction du prix en temps réel
#### Visualisation graphique :
Point rouge: maisons du dataset

Point bleu: maison saisie par l’utilisateur

L’application permet de comparer visuellement la prédiction utilisateur avec les données historiques du
dataset.

## Structure du projet
```text 
PriceVision/
│
├── Data/
│ └── House_prices_kaggle.csv
│
├── notebooks/
│ └── exploration.ipynb
│
├── app/
│ └── app.py
│
├── model.pkl
├── requirements.txt
└── README.md

Lancer le projet en local

Créer un environnement virtuel

python3 -m venv venv

source venv/bin/activate

Installer les dépendances
pip install -r requirements.txt

Lancer l’application Streamlit
streamlit run app/app.py

Technologies utilisées
Python 3

Pandas / NumPy

Scikit-learn

Matplotlib

Streamlit

