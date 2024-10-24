# Packages
import os
from pathlib import Path
import re
import shutil
import zipfile
import rasterio
from rasterio.plot import show
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import seaborn as sns
import calendar
import numpy as np
import ast
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.ensemble import RandomForestClassifier


def extract_jp2_files(input_folder, output_folder, resolution):
    """
Extracts all the JP2 files according to the image resolution you want from 
the input Zip files to the specified output folder and into folders named 
with the common root of the JP2 files they contain 
- param input_folder: the Zip folder downloaded from the Copernicus site 
- param output_folder: the folder where the folders containing the JP2 files are extracted 
- param resolution: the image resolution you want for the JP2 files 
to extract among the available resolutions (10{m},20{m},60{m})
    """
    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.zip'):
                # Ouverture du fichier zip
                with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                    # Recherche du dossier IMG_DATA
                    img_data_folder = next(
                        (item for item in zip_ref.namelist() if 'IMG_DATA' in item), None)
                    if img_data_folder:
                        # Recherche et copie des fichiers avec la résolution donnée
                        jp2_files = [f for f in zip_ref.namelist(
                        ) if f'{img_data_folder}R{resolution}m/' in f and f.endswith(
                            f'{resolution}m.jp2')]
                        if jp2_files:
                            # Création du dossier de sortie avec le nom commun des fichiers JP2
                            common_name = os.path.commonprefix(
                                [os.path.basename(j) for j in jp2_files]
                            ).rstrip(f'{resolution}m.jp2').replace('/', '')
                            folder_path = os.path.join(
                                output_folder, f'{common_name}R{resolution}m')
                            os.makedirs(folder_path, exist_ok=True)
                            # Copie des fichiers JP2 dans le dossier de sortie
                            for jp2_file in jp2_files:
                                with zip_ref.open(jp2_file, 'r') as f_in, open(
                                        os.path.join(folder_path, os.path.basename(jp2_file)), 'wb'
                                ) as f_out:
                                    shutil.copyfileobj(f_in, f_out)


def convert_jp2_to_tif(input_folder):
    """
Converts all JP2 files in the input folder into a single TIF file,
stored in the input folder. JP2 files are deleted after conversion.
Empty folders are also deleted after conversion.
- param input_folder: the folder containing the JP2 files
    """
    # Parcours des dossiers du input_folder
    for folder_path in os.scandir(input_folder):
        if not folder_path.is_dir():
            continue
        jp2_files = sorted(f.path for f in os.scandir(folder_path.path) if f.name.endswith('.jp2'))
        if not jp2_files:
            print(f'Pas de fichiers JP2 trouvés dans {folder_path}.')
            continue
        # Création du nom du fichier TIF
        tif_file = os.path.join(input_folder, f'{folder_path.name}.tif')
        band_names_map = {
            'AOT': 'Aerosol Optical Thickness', 'B01': 'Blue',
            'B02': 'Green', 'B03': 'Red', 'B04': 'Red Edge 1',
            'B05': 'Red Edge 2', 'B06': 'Red Edge 3', 'B07': 'NIR',
            'B08': 'NIR Narrow', 'B8A': 'NIR Wide', 'B09': 'Water Vapor',
            'B11': 'SWIR 1', 'B12': 'SWIR 2', 'SCL': 'Scene Classification Map',
            'TCI': 'True Color', 'WVP': 'Water Vapor'
        }
        band_name = [os.path.basename(f).split('_')[2] for f in jp2_files]
        desc_band_names = []
        with rasterio.open(jp2_files[0]) as src:
            meta = src.meta.copy()
            #mise à jour des métadonnées (exemple: -999 pour nodata
            #afin d'éviter les divisions par zéro)
            meta.update(count=len(jp2_files), driver='GTiff', tiled=True)
            with rasterio.open(tif_file, 'w', **meta) as dst:
                for i, file_path in enumerate(jp2_files, start=1):
                    desc_band_names.append(band_names_map[band_name[i-1]])
                    with rasterio.open(file_path) as src2:
                        # Mettre à jour la description des données et les tags des bandes
                        dst.update_tags(bands=band_name)
                        dst.write(src2.read(1), i)
                dst.update_tags(descriptions=desc_band_names)
        # Suppression des fichiers JP2 et du dossier
        shutil.rmtree(folder_path.path)


# Fonction pour lire les données d'une image
def read_data(filepath):
    with rasterio.open(filepath) as src:
        # Lecture de chaque bande de l'image, sauf la dernière (étiquettes)
        bands = [src.read(i, out_shape=(src.height // 4, src.width // 4)
                          ) for i in range(1, src.count)]
        # Conversion des bandes en un tableau numpy
        X = np.stack(bands, axis=-1)
        # Lecture de la dernière bande (étiquettes)
        y = src.read(src.count, out_shape=(src.height // 4, src.width // 4))
    return X, y