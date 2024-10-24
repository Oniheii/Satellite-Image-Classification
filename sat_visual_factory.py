# Packages
import os
import rasterio
from rasterio.plot import show
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import seaborn as sns
import calendar
import numpy as np


# recuparation de date à partir du nom du fichier
def date_info(filename):
    bn = os.path.basename(filename)
    ds = bn.split("_")[1].split("T")[0]
    return int(ds[:4]), int(ds[4:6])


# matchin de date et saison
def season(month):
    if month in range(3, 6): return "Printemps"
    if month in range(6, 9): return "Été"
    if month in range(9, 12): return "Automne"
    return "Hiver"


# Comparaison visuelle des changements spectraux pour chaque paire d'images
def compare_images(im1, im2, key_name=None):
    with rasterio.open(im1) as s1, rasterio.open(im2) as s2:
        image1, image2 = s1.read(1), s2.read(1)
        year1, month1 = date_info(im1)
        year2, month2 = date_info(im2)
        s1, s2 = season(month1), season(month2)
        # Création de la figure
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # Cartes des deux images
        axs[0, 0].imshow(image1, cmap='RdYlGn')  # cmap='gray'
        axs[0, 0].set_title(f"{key_name} {calendar.month_name[month1]} {year1}")
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(image2, cmap='RdYlGn')  # cmap='gray'
        axs[0, 1].set_title(f"{key_name} {calendar.month_name[month2]} {year2}")
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        # Différence spectrale
        diff = image2 - image1
        axs[1, 0].imshow(diff, cmap='bwr')  # cmap='bwr' #'RdYlGn'
        axs[1, 0].set_title(
            f"Diff spectrale entre {s1} et {s2}"
        )
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 0].spines['top'].set_visible(False)
        axs[1, 0].spines['right'].set_visible(False)
        axs[1, 0].spines['bottom'].set_visible(False)
        axs[1, 0].spines['left'].set_visible(False)
        # Profil spectral
        col = int(image1.shape[1] / 2)
        axs[1, 1].plot(image2[:, col], color='red',
                       label=f"{calendar.month_name[month2]} {year2}")
        axs[1, 1].plot(image1[:, col], color='blue',
                       label=f"{calendar.month_name[month1]} {year1}")
        axs[1, 1].set_title("Profil spectral")
        axs[1, 1].set_xlabel('Lignes')
        axs[1, 1].set_ylabel('Réflectance')
        axs[1, 1].legend(loc='upper right')
        axs[1, 1].spines['top'].set_visible(False)
        axs[1, 1].spines['right'].set_visible(False)
        # plt.tight_layout()
        return (fig)
