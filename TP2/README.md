# Projet 1

## Setup

Tous les packets requis sont rassemblés dans le fichier `requirements.txt`, et peuvent être installés en utilisant la commande suivante:

```
$ pip install -r requirements.txt
```

## Utilisation

#### Question 1

Le fichier `histogram.py` peut prendre plusieurs arguments à l'appel:
* path: Chemin vers le dossier contenant les données (obligatoire)
* --bins: Nombre de subdivisions dans l'histogramme
* --log_scale: Affichage des données en logarithme
* --remove_black: Retire les x% de pixels les plus sombres et affiche un second histogramme

Exemple de commande correcte:

```
$ python histogram.py Data_TP2/ --bins 50 --log_scale --remove_black
```
