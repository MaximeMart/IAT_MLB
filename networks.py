import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, distJoueur: int, distBalle: int, sensInvader: int, strStateBullet: str, na: int): #j'ai pas mis nf le nombre de plans car je voyais pas l'éqsuivalent dans notre cas
        """À MODIFIER QUAND NÉCESSAIRE.
        Ce constructeur crée une instance de réseau de neurones de type Multi Layer Perceptron (MLP).
        L'architecture choisie doit être choisie de façon à capter toute la complexité du problème
        sans pour autant devenir intraitable (trop de paramètres d'apprentissages). 

        :param na: Le nombre d'actions 
        :type na: int
        """
        if (strStateBullet=="fire") :
            stateBullet = 1
        else:
            stateBullet = 0
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(distJoueur*distBalle*sensInvader*stateBullet, 32),
            nn.ReLU(),
            nn.Linear(32, na),
        )

    def forward(self, x):
        """Cette fonction réalise un passage dans le réseau de neurones. 

        :param x: L'état
        :return: Le vecteur de valeurs d'actions (une valeur par action)
        """
        x = self.flatten(x)
        qvalues = self.layers(x)
        return qvalues

class CNN(nn.Module):
    def __init__(self, distJoueur: int, distBalle: int, sensInvader: int, stateBullet: int, na: int):
        """À MODIFIER QUAND NÉCESSAIRE.
        Ce constructeur crée une instance de réseau de neurones convolutif (CNN).
        L'architecture choisie doit être choisie de façon à capter toute la complexité du problème
        sans pour autant devenir intraitable (trop de paramètres d'apprentissages). 

        :param na: Le nombre d'actions 
        :type na: int
        """
        super(CNN, self).__init__()
        #pas compris donc j'y touche pas trop pour l'instant
        self.layers = nn.Sequential(
            #nn.Conv2d(nf, 32, 3, stride=1, padding="same", bias=True), #à revoir pour adapter c ar pas de nf dans les paramêtre d'entrée
            #nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding="same", bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(distJoueur*distBalle*sensInvader*stateBullet * 32, 120),
            nn.Linear(120, na)
        )

        self.layers.apply(weights_init)

    def forward(self, x):
        """Cette fonction réalise un passage dans le réseau de neurones. 

        :param x: L'état
        :return: Le vecteur de valeurs d'actions (une valeur par action)
        """
        qvalues = self.layers(x)
        return qvalues

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)