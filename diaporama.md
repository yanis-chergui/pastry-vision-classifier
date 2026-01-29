---
jupytext:
  notebook_metadata_filter: rise
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  auto_select: first
  autolaunch: false
  backimage: fond.png
  centered: false
  controls: false
  enable_chalkboard: true
  height: 100%
  margin: 0
  maxScale: 1
  minScale: 1
  scroll: true
  slideNumber: true
  start_slideshow_at: selected
  transition: none
  width: 90%
---

+++ {"slideshow": {"slide_type": "slide"}}

# Jeux de données : Madeleines et Macarons

- Binôme: Alexis Bezos, Yanis Chergui, Jean Dersoir
- Adresses mails: alexis.bezos@universite-paris-saclay.fr, yanis.chergui@universite-paris-saclay.fr, jean.dersoir@universite-paris-saclay.fr
- [Dépôt GitLab](https://gitlab.dsi.universite-paris-saclay.fr/xxx.yyy/Semaine8/)

+++ {"slideshow": {"slide_type": "slide"}}

## Jeu de données

:::{admonition} Consignes
:class: dropdown

Description du jeu de données et motivation: en quoi est-ce intéressant ?
Voir la [feuille 1 sur le chargement du jeu de données](1_jeu_de_donnees.md)

:::

```{code-cell} ipython3
extension = 'jpg' 
```

```{code-cell} ipython3
import os, re
from glob import glob as ls
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image
%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings("ignore")
from sys import path

from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *

from intro_science_donnees import *  #les librairies classiques pour ISD (numpy, PIL...) y sont incluses
# Configuration intégration dans Jupyter
%matplotlib inline

dataset_dir = 'data'
images_a = load_images(dataset_dir, f"a*.{extension}")
images_b = load_images(dataset_dir, f"b*.{extension}")
images = pd.concat([images_a, images_b])
image_grid(images, titles=images.index)
```

```{code-cell} ipython3
assert len(images_a) >= 10, "Vous devez avoir au moins 10 images dans la classe A."

assert len(images_b) >= 10, "Vous devez avoir au moins 10 images dans la classe B."


aspect_ratios = [ image.height / image.width for image in images]
assert abs ( max(aspect_ratios) - min(aspect_ratios) ) <= 0.1, \
    "Vos images n'ont pas toutes le même rapport hauteur / largeur (aspect ratio)"


from logging import warning
for name,image in images.items():
    if image.width > 256: warning(f"Image {name} trop large")
    if image.height > 256: warning(f"Image {name} trop haute")
```

+++ {"slideshow": {"slide_type": "slide"}}

## Prétraitement 

:::{admonition} Consignes
:class: dropdown

Décrivez les étapes de prétraitement effectuées : recadrage; réduction
de la résolution + lissage; choix d'attributs etc.  Voir la [feuille 2
sur une première analyse des données](2_premiere_analyse_ACP.md) et la
[feuille 3 sur l'extraction des
attributs](3_extraction_d_attributs.md)

:::

```{code-cell} ipython3
"""
from PIL import Image

img = []
def croppedsqrd():
    for image in images:
        im1 = image
        top = -50
        right = 256
        left = 0
        bottom = 206
        im1 = im1.crop((left, top, right, bottom))
        img.append(im1)
        #im1.save('IntroScienceDonnees/Semaine7/data')
croppedsqrd()
image_grid(img)
"""
```

```{code-cell} ipython3
def difference_filter(img:Image.Image) -> np.ndarray:
    """Extract a numpy array D = R-(G+B)/2 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    D = R - (G + B) / 2
    return D
```

```{code-cell} ipython3
def value_filter(img:Image.Image) -> np.ndarray:
    """Extract a numpy array V = (R+G+B)/3 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    V = (R + G + B) / 3
    return V
```

```{code-cell} ipython3
def foreground_redness_filter(
    img: Image.Image, theta: float = 2 / 3
) -> np.ndarray:
    """Extract a numpy array with True as foreground
    and False as background from a PIL image.
    Parameter theta is a relative binarization threshold."""
    D = difference_filter(img)
    V = value_filter(img)
    F0 = np.maximum(D, V)
    threshold = theta * (np.max(F0) - np.min(F0))
    F = F0 > threshold
    return F
```

```{code-cell} ipython3
def my_filter(imag):
    l = [foreground_redness_filter(img, theta = .8) for img in imag]
    return l

X = my_filter(images)
image_grid(X)
```

```{code-cell} ipython3
def perimeter(img: Image.Image) -> float:
    """Extract the scalar value perimeter from a PIL image."""
    C = contours(img)
    return np.count_nonzero(C)
```

```{code-cell} ipython3
def contours(img: Image.Image) -> np.ndarray:
    M = np.array(img)
    contour_horizontal = np.abs(M[:-1, 1:] ^ M[:-1, 0:-1])
    contour_vertical = np.abs(M[1:, :-1] ^ M[0:-1, :-1])
    return contour_horizontal + contour_vertical
```

```{code-cell} ipython3
def yellowness_filter(img:[Image.Image, np.ndarray]) -> np.ndarray:
    """Return a grey-level image measuring the yellowness of each pixel."""
    # Remplacer la ligne suivante par le code adéquat
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    return (R + G - B).mean()

def find_black_pixels(img: Image.Image) -> tuple[int, int]:
    """
    Trouve les indices du premier et du dernier pixel noir sur la ligne du milieu en largeur de l'image.
    
    Paramètres:
    img (PIL.Image.Image) - Image à analyser.
    
    Retourne:
    tuple[int, int] - Tuple contenant l'indice du premier et du dernier pixel noir.
    """
    # Convertir l'image en tableau numpy
    pixels = np.array(img)
    
    # Récupérer la ligne du milieu en largeur
    middle_row = pixels[pixels.shape[0] // 2]
    
    # Trouver les indices du premier et du dernier pixel noir
    first_black = np.where(middle_row != 255)[0][0]
    last_black = np.where(middle_row != 255)[0][-1]
    
    return last_black-first_black

def standardize_df(df):
    """Standardize all the columns except the last one (target values)."""
    df_scaled = (df - df.mean()) / df.std()
    df_scaled.iloc[:, -1] = df.iloc[:, -1]
    return df_scaled
```

```{code-cell} ipython3
df = pd.DataFrame({
        'yellowness':    images.apply(yellowness_filter),  
        'find_black_pixels':    images.apply(find_black_pixels),  
        'class':      images.index.map(lambda name: 1 if name[0] == 'a' else -1),
})
df
```

```{code-cell} ipython3
new_df=standardize_df(df)
new_df
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Charger ta base de données d'images
y = new_df['yellowness'].tolist()  # Remplace [...] par ta liste de noms d'images
x = new_df.index.tolist()     # Remplace [...] par ta liste de valeurs de yellowness
x = [image[:-4] if int(image[1:3]) > 9 else image[0] + image[2:-4] for image in x]
# Création du graphique à barres
plt.bar(x, y)

# Ajout de titres et d'étiquettes
plt.title('Exemple de graphique à barres')
plt.xlabel('Images')
plt.ylabel('Yellowness')

# Affichage du graphique
plt.show() 
```

```{code-cell} ipython3
y = new_df['find_black_pixels'].tolist()  # Remplace [...] par ta liste de noms d'images
x = new_df.index.tolist()     # Remplace [...] par ta liste de valeurs de yellowness
x = [image[:-4] if int(image[1:3]) > 9 else image[0] + image[2:-4] for image in x]
# Création du graphique à barres
plt.bar(x, y)

# Ajout de titres et d'étiquettes
plt.title('Exemple de graphique à barres')
plt.xlabel('Images')

plt.ylabel('find_black_pixels')

# Affichage du graphique
plt.show() 
```

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = new_df
df = pd.DataFrame(data)

# Création du heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)

# Ajout de titres et d'étiquettes
plt.title('Corrélation entre find_black_pixels et yellowness')
plt.xlabel('Attribut')
plt.ylabel('Attribut')

# Affichage du heatmap
plt.show()
```

```{code-cell} ipython3
sns.scatterplot(data=new_df,x="find_black_pixels",y="yellowness",hue ='class')
```

tu peux potentiellement utiliser la forme faite avec les pixels noirs pour calculer l'élongation

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Divisez vos données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(new_df[['find_black_pixels', 'yellowness']], new_df['class'], test_size=0.2, random_state=42)

# Entraînez un modèle KNN avec un nombre de voisins (k) donné
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Tracer le nuage de points avec la frontière de décision
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Tracer les limites de la figure
x_min, x_max = X_train['find_black_pixels'].min() - 1, X_train['find_black_pixels'].max() + 1
y_min, y_max = X_train['yellowness'].min() - 1, X_train['yellowness'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Tracer le résultat
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Tracer les points d'entraînement
plt.scatter(X_train['find_black_pixels'], X_train['yellowness'], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Nombre de pixels noirs')
plt.ylabel('Teinte de jaune')
plt.title('Nuage de points avec frontière de décision (KNN)')
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Classificateurs favoris et visualisation des données

:::{admonition} Consignes
:class: dropdown

Quel classificateur avez vous choisi ? Quels sont les résultats et les
taux d'erreur ? Montrez vos résultats! Après le cours en Semaine 8,
faites la [feuille 4 sur les classificateurs](4_classificateurs.md)

:::

```{code-cell} ipython3
def split_data(X, Y, verbose=True, seed=0):
    """Make a 50/50 training/test data split (stratified).
    Return the indices of the split train_idx and test_idx."""
    SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    ((train_index, test_index),) = SSS.split(X, Y)
    if verbose:
        print("TRAIN:", train_index, "TEST:", test_index)
    return (train_index, test_index)
def error_rate(solutions: pd.Series, predictions: pd.Series) -> Any:
    """
    Return the error rate between two vectors.
    """
    return (solutions != predictions).mean()
```

```{code-cell} ipython3
X = new_df[['yellowness', 'find_black_pixels']]  
Y = new_df['class']
# Partition des images
train_index, test_index = split_data(X, Y, seed=3)

# Partition de la table des attributs
Xtrain = X.iloc[train_index]
Xtest = X.iloc[test_index]

# Partition de la table des étiquettes
Ytrain = Y.iloc[train_index]
Ytest = Y.iloc[test_index]
```

```{code-cell} ipython3

#définition du classificateur, ici on l'appelle classifier
# on choisit k=1
classifier = KNeighborsClassifier(n_neighbors=1)
# on l'ajuste aux données d'entrsainement
classifier.fit(Xtrain, Ytrain) 
# on calcule ensuite le taux d'erreur lors de l'entrainement et pour le test
Ytrain_predicted = classifier.predict(Xtrain)
Ytest_predicted = classifier.predict(Xtest)
# la fonction error_rate devrait etre présente dans votre utilities.py (TP3), sinon ajoutez-la
e_tr = error_rate(Ytrain, Ytrain_predicted)
e_te = error_rate(Ytest, Ytest_predicted)

print("Classificateur: 1 Nearest Neighbor")
print("Training error:", e_tr)
print("Test error:", e_te)
```

```{code-cell} ipython3
from sklearn.neighbors import RadiusNeighborsClassifier

# Créer le classificateur
classifier = RadiusNeighborsClassifier(radius=2.0)

# Ajuster le classificateur aux données d'entraînement
classifier.fit(Xtrain, Ytrain )

# Calculer les taux d'erreur pour l'entraînement et le test
e_tr = 1 - classifier.score(Xtrain, Ytrain)
e_te = 1 - classifier.score(Xtest, Ytest)

# Afficher les résultats
print("Classificateur: Parzen Window")
print("Training error:", e_tr)
print("Test error:", e_te)
```

```{code-cell} ipython3
from sklearn import tree

# Créer le classificateur
classifier = tree.DecisionTreeClassifier()

# Ajuster le classificateur aux données d'entraînement
classifier.fit(Xtrain, Ytrain)

# Calculer les prédictions du classifieur sur les ensembles d'entraînement et de test
Ytrain_pred = classifier.predict(Xtrain)
Ytest_pred = classifier.predict(Xtest)

# Calculer les taux d'erreurs
e_tr = 1 - classifier.score(Xtrain, Ytrain)
e_te = 1 - classifier.score(Xtest, Ytest)

# Afficher les résultats
print("Classificateur: Arbre de décision")
print("Training error:", e_tr)
print("Test error:", e_te)
```

```{code-cell} ipython3
from sklearn.linear_model import Perceptron

# Définition du modèle de classificateur
classifier = Perceptron()

# Ajustement aux données d'entraînement
classifier.fit(Xtrain, Ytrain)

# Calcul du taux d'erreur lors de l'entraînement et pour le test
e_tr = 1 - classifier.score(Xtrain, Ytrain)
e_te = 1 - classifier.score(Xtest, Ytest)

# Affichage des taux d'erreurs
print("Classificateur: Perceptron")
print("Training error:", e_tr)
print("Test error:", e_te)
```

## Interprétations

:::{admonition} Consignes
:class: dropdown

Pouvez vous commenter vos taux d'erreur ? Le jeu de données était il
trop simple ou trop compliqué ? Une photo en particulier était elle
biaisée ? Etc.

:::

+++ {"slideshow": {"slide_type": "slide"}}

## Discussion et conclusion

:::{admonition} Consignes
:class: dropdown

Vous pourriez parler *par exemple* des biais potentiels de vos
données, de l'utilisation d'un tel projet dans la vraie vie, des
difficultés rencontrées et de comment vous les avez surmontées (ou
pas).

:::
