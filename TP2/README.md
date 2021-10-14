# Projet 1

## Setup

Tous les packets requis sont rassemblés dans le fichier `requirements.txt`, et peuvent être installés en utilisant la commande suivante:

```
$ pip install -r requirements.txt
```

## Utilisation

### Question 1 - Histogramme conjoint

Le fichier `histogram.py` peut prendre plusieurs arguments à l'appel:
* {+ path +}: Chemin vers le dossier contenant les données (obligatoire)
* --bins: Nombre de subdivisions dans l'histogramme
* --log_scale: Affichage des données en logarithme
* --remove_black: Retire les x% de pixels les plus sombres et affiche un second histogramme

Exemple de commande correcte:

```
$ python histogram.py Data_TP2/ --bins 50 --log_scale --remove_black 95
```

### Question 2 - Critère de similarité

Les différentes fonctions sont implémentées dans le fichier `similarity.py`. Celles-ci sont automatiquement appelées avec le fichier `histogram.py`et produisent un résultat dans le terminal. Par exemple:

````
Paire 6:
Histogram sum:                381924.0
Number of pixels:             381924
SSD:                          42510000
Correlation:                  0.7802495447254987
IM:                           11.320345378217311
````
### Question 3 - Transformations spatiales

Les fonctions et classe sont implémentées dans le fichier `transformation.py`. Les arguments sont:
* param: Les paramètres des transformations. (Dans l'ordre [d, theta, omega, phi, p, q, r])
* --transformation: Rigide ou similitude.

Par exemple:
````
$ python transformation2.py 0.5 45 45 45 0 0 0 --transformation similitude
````
