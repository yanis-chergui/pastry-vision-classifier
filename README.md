# 🥐 Pastry Vision: Macaron vs. Madeleine

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Computer Vision](https://img.shields.io/badge/Topic-Computer_Vision-purple?style=flat&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/Models-KNN_|_Tree_|_Perceptron-orange?style=flat)

> **"Is it a Macaron or a Madeleine?"**
>
> Ce projet de **Computer Vision** explore les fondamentaux de la classification d'images en entraînant plusieurs modèles à distinguer deux pâtisseries françaises.

## 🎯 Contexte & Objectif

L'objectif est de développer un pipeline complet de reconnaissance d'images, depuis le traitement des pixels bruts jusqu'à la classification binaire.

Bien que le sujet soit léger (pâtisseries 🍰), les concepts sont identiques à ceux de **l'analyse vidéo sportive** (reconnaissance de maillots, tracking de ballon).

## ⚽ Lien avec l'Analytique Sportive (Hackathon FCB)

Ce projet démontre les compétences techniques pour traiter des données visuelles :

1.  **Traitement du Signal :** Transformation d'images brutes en vecteurs (Feature Extraction), similaire à l'extraction de positions joueurs.
2.  **Classification Supervisée :** Comparaison de modèles pour distinguer des classes (ex: "Passe" vs "Tir", ici "Macaron" vs "Madeleine").
3.  **Évaluation :** Analyse rigoureuse des erreurs (Training vs Test Error).

## ⚙️ Pipeline Technique

### 1. Prétraitement (Image Processing)
Le script `utilities.py` gère la manipulation d'images :
* **Cropping :** Recadrage automatique.
* **Redimensionnement :** Standardisation $32 \times 32$ pixels.
* **Vectorisation :** Aplatissement (Flattening) des matrices RGB.

### 2. Modélisation (Machine Learning)
Comparaison de trois approches via `Scikit-learn` :
* **KNN (K-Nearest Neighbors)**
* **Perceptron** (Modèle linéaire)
* **Arbre de Décision** (Non-linéaire)

## 📊 Résultats

Le **Perceptron** montre les limites des modèles linéaires sur des images brutes (sensible au bruit). Les **Arbres de Décision** et le **KNN** offrent une meilleure capacité à capturer les motifs visuels complexes.

## 🛠️ Installation

1.  Cloner le projet :
    ```bash
    git clone [https://github.com/yanis-chergui/pastry-vision-classifier.git](https://github.com/yanis-chergui/pastry-vision-classifier.git)
    ```
2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
3.  Lancer la présentation (nécessite Jupytext) :
    ```bash
    jupyter notebook 0_diaporama.md
    ```

---
*Projet réalisé dans le cadre du cursus Science des Données à l'Université Paris-Saclay.*
