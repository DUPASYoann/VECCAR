# VECCAR : Vectorisation de carte raster
###### Projet Recherche et Developpement

VECCAR est un projet visant à élaborer une solution permettant la vectorisation automatique et intelligente de documents rasterisés.

## Quoi de neuf ?

* Adaptation du réseau DeepLab
* Script pour le jeu de donnée "cartographie edition 1960"
* ajout de la loss function "cross entropy weigted" (poids à définir manuellement)

## Guide d'installation

### Requirment

* Linux OS
* CUDA (voir https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)
* torch (voir https://pytorch.org/get-started/locally/)
* Detectronv2 (voir https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
* opencv-python
* Tensorboard (pour la visualisation)

#### __ATTENTION AUX VERSIONS__

Vérifiez bien les compatibilitées des versions entre CUDA Torch et Detectronv2.

#### Si CUDA 10.1 d'installé

pip install -r requirements.txt

## Wiki


Encadrants : Dominique ANDRIEUX ; Romain RAVEAUX ; Barthélémy SERRES

Etudiant : Yoann DUPAS <yoann.dupas@outlook.fr>
