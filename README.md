# Projet Deep-Q Poly#

Implémentation d'une solution pour placer des bâtiment sur une carte (Google Hashcode 2018) grâce à l'apprentissage par renforcement.

Ce dépôt contient les fichiers Python pour l'apprentissage et le tests des différents modèles. Il contient aussi l'arbitre (dans referee), l'environement développé sur Gym (dans gym-env) et les données de Google (dans data) et les petites cartes (dans data_small).

# Dépendances

- [gym](https://github.com/openai/gym#installation)
- [keras-rl](https://github.com/keras-rl/keras-rl)
- [tensorflow](https://github.com/tensorflow/tensorflow)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [imageio](https://imageio.readthedocs.io/en/latest/installation.html)

Il est possible d'installer les dépendances directement :
```
pip install -r requirements.txt
```

# Usage

Pour lancer l'apprentissage sur le modèle par renforcement naïf :

```
python3 main.py
```


Pour lancer l'apprentissage sur le modèle par renforcement profond naïf :
```
python3 main_RL_1D.py 
```


Pour lancer l'apprentissage sur le modèle par renforcement profond avancé :
```
python3 main_RL_advanced.py 
```


Pour lancer l'apprentissage sur le modèle par renforcement profond avancé :
```
python3 main_RL_advanced.py 
```


Pour lancer les tests sur le modèle par renforcement profond avancé :
```
python3 main_RL_advanced_test.py 
```

Le fichier main_RL_TwoInputs.py contient le test de l'agent à deux entrées.

# L'équipe

Simon LASSALLE
Pierre SAVATTE

# Commanditaires

Matthieu PERREIRA DA SILVA
Erwan DAVID
