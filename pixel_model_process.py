# packages
import os
from pathlib import Path
import re
import shutil
import zipfile
import rasterio
from rasterio.plot import show
from matplotlib  import colors
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


# recupérer les valeurs pour la bande NIR disponible
def get_nir_band(band_dict):
    """
    Obtient la bande NIR (B08, B8A ou B07) dans un dictionnaire de bandes.
    :param band_dict: Dictionnaire de bandes
    :return: La bande NIR et le nom de la bande
    """
    if 'B08' in band_dict:
        nir = band_dict['B08']
        nir_band = 'B08'
    elif 'B8A' in band_dict:
        nir = band_dict['B8A']
        nir_band = 'B8A'
    elif 'B07' in band_dict:
        nir = band_dict['B07']
        nir_band = 'B07'
    else:
        raise ValueError("Unable to find NIR band (B08, B8A or B07)")
    # Afficher la bande NIR utilisée
    # print(f"NIR band used: {nir_band}")
    return nir, nir_band


# labeliser les pixels dans les images de cartes
def labeling_pixel(input_folder, output_folder=None, class_names=None,
                   NDVI_threshold=0.2, SW1_threshold=0.5, SW2_threshold=0.3):
    # Sélectionner les bandes nécessaires au calcul des indices
    select_bands_tags = ['B04', 'B07', 'B08', 'B8A', 'B05', 'B06']
    # select_bands_desc = ['Red','NIR','Red Edge','NDVI','SW1','SW2']
    for f in os.listdir(input_folder):
        with rasterio.open(os.path.join(input_folder, f)) as src:
            # Récupérer les bandes sélectionnées
            indexed_bands = zip(
                ast.literal_eval(src.tags()['bands']), src.indexes)
            select_band_dict = {tag: src.read(index) for tag,
            index in indexed_bands if tag in select_bands_tags}
            # Bandes pour calculs d'indices de végétation
            red = select_band_dict['B04']
            nir, nir_band = get_nir_band(select_band_dict)
            red_edge_05, red_edge_06 = select_band_dict['B05'], select_band_dict['B06']
            # Calcul des indices
            ndvi = (nir - red) / (nir + red + 0.0001)
            sw1 = red_edge_05 / (nir + 0.0001)
            sw2 = red_edge_06 / (nir + 0.0001)
            # Pixel labeling using decision tree rules
            labeled = np.zeros_like(red, dtype=np.uint8)
            # Cropland
            labeled[(ndvi < NDVI_threshold) & (sw1 < SW1_threshold)] = 5
            # Grassland/Herbaceous
            labeled[(ndvi >= NDVI_threshold) & (sw1 < SW1_threshold) & (sw2 < SW2_threshold)] = 0
            # Broadleafed woody vegetation
            labeled[(ndvi >= NDVI_threshold) & (sw1 >= SW1_threshold) & (sw2 < SW2_threshold)] = 1
            # Coniferous woody vegetation
            labeled[(ndvi >= NDVI_threshold) & (sw1 < SW1_threshold) & (sw2 >= SW2_threshold)] = 2
            # Built-up
            labeled[(ndvi >= NDVI_threshold) & (sw1 >= SW1_threshold) & (sw2 >= SW2_threshold)] = 3
            # Water
            labeled[(ndvi < NDVI_threshold) & (sw1 >= SW1_threshold)] = 4
            # tags = src.tags()
            # le dossier de sortie
            os.makedirs(output_folder, exist_ok=True) if output_folder else None
            out_meta = src.meta.copy()
            out_meta.update({'count': len(src.indexes) + 1,
                             'description': "Pixel-wise classification with NDVI and SWIR indices"
                             })
            out_file = os.path.join(output_folder, f[:-4] + "_labeled.tif")
            with rasterio.open(out_file, 'w', **out_meta) as dst:
                for i, (tag, band) in enumerate(select_band_dict.items()):
                    dst.write(band, i + 1)
                # Ajout de la nouvelle bande d'étiquettes
                dst.write(labeled, len(src.indexes) + 1)
                # Copier les tags de src
                tags = src.tags()
                # Mettre à jour les tags pour la nouvelle bande d'étiquettes
                tags = {str(k): v for k, v in tags.items()}
                dst.update_tags(**tags)


def create_image_dict(input_folder):
    """
    Crée un dictionnaire de paires d'images qui se chevauchent et de
    fichiers uniques qui n'ont pas de paire.
    """
    image_pairs = {}
    single_images = {}
    for file in os.listdir(input_folder):
        if file.endswith(".tif") or file.endswith("_labeled.tif"):
            location = file[:6]
            if location in image_pairs:
                image_pairs[location].append(os.path.join(input_folder, file))
            elif location in single_images:
                image_pairs[location] = [(single_images[location],
                                          os.path.join(input_folder, file))]
                del single_images[location]
            else:
                single_images[location] = os.path.join(input_folder, file)
    return image_pairs, single_images


# Fonction pour créer des listes de fichiers d'entraînement, de test et de validation
def create_train_test_lists(input_folder):
    # Création d'un dictionnaire d'images
    image_pairs, single_images = create_image_dict(input_folder)
    # Ajout de la première image de chaque paire à la liste d'entraînement
    train_files = [pair[0][0] for pair in image_pairs.values()]
    # Ajout de la deuxième image de chaque paire à la liste de test
    test_files = [pair[0][1] for pair in image_pairs.values()]
    # Ajout des images seules à la liste de validation
    validation_files = list(single_images.values())
    return train_files, test_files, validation_files


# Fonction pour entraîner un modèle Decision Tree
def train_dt(X_train, y_train, max_depth=None, random_state=42):
    # Initialisation du modèle
    dt = DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state
    )
    # Entraînement du modèle sur les données d'entraînement
    dt.fit(X_train.reshape(-1, X_train.shape[-1]), y_train.ravel())
    return dt


# Fonction pour tester un modèle Decision Tree
def test_dt(dt, X_test, y_test):
    # Prédiction des étiquettes à partir des données de test
    y_pred = dt.predict(X_test.reshape(-1, X_test.shape[-1]))
    # Calcul des métriques de précision, rappel, F-score et support
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test.ravel(), y_pred, average='weighted'
    )
    # Affichage des métriques
    print(f"Accuracy: {accuracy_score(y_test.ravel(), y_pred)}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-score: {fscore}")
    print(f"Support: {support}")

    def cross_validation_dt(input_files, class_names, n_splits=2,
                            n_estimators=100, max_depth=None,
                            max_features=None, random_state=42):
        # Création des listes d'entraînement, de test et de validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Initialisation des scores de validation croisée
        scores = {'accuracy': [], 'precision': [],
                  'recall': [], 'f1-score': []}

        # Calcul de la matrice de confusion moyenne
        conf_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

        for train_index, test_index in kf.split(input_files):
            # Séparation des données en ensembles d'entraînement et de test
            X_train, y_train = [], []
            for file_index in train_index:
                X, y = read_data(input_files[file_index])
                X_train.append(X[:, :, 0:15])
                y_train.append(y)

            # Fusion des ensembles d'entraînement et de validation
            X_test, y_test = read_data(input_files[test_index[0]])
            X_test = X_test[:, :, 0:15]
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)

            # Entraînement du modèle
            dt = DecisionTreeClassifier(max_depth=max_depth,
                                        max_features=max_features,
                                        random_state=random_state)
            dt.fit(X_train.reshape(-1, X_train.shape[-1]), y_train.ravel())

            # Prédiction des étiquettes pour l'ensemble de test
            y_pred = dt.predict(X_test.reshape(-1, X_test.shape[-1]))

            # Calcul de la matrice de confusion
            conf_matrix += confusion_matrix(
                y_test.ravel(), y_pred, labels=list(class_names.keys()))

            # Calcul des scores pour chaque fold
            accuracy, precision, recall, f1_score = precision_recall_fscore_support(
                y_test.ravel(), y_pred, labels=list(class_names.values()),
                average='weighted', zero_division=0)[:4]

            # Ajout des scores pour le fold courant
            if (accuracy is not None and
                    precision is not None and
                    recall is not None and
                    f1_score is not None):
                scores['accuracy'].append(accuracy)
                scores['precision'].append(precision)
                scores['recall'].append(recall)
                scores['f1-score'].append(f1_score)

        # Calcul de la matrice de confusion moyenne
        conf_matrix = conf_matrix / np.sum(conf_matrix) * 100

        # Affichage graphique de la matrice de confusion moyenne
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.4)
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.1f',
                    xticklabels=class_names.values(),
                    yticklabels=class_names.values())
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité Terrain')
        plt.title('Matrice de confusion moyenne')
        plt.savefig('visualisation/confusion_matrix_dt.png')

        # Calcul des scores moyens de validation croisée
        mean_scores = {k: np.mean(scores[k]) for k in scores.keys()}

        # Affichage des scores moyens de validation croisée
        mean_accuracy = np.mean(scores['accuracy'])
        mean_precision = np.mean(scores['precision'])
        mean_recall = np.mean(scores['recall'])
        mean_f1_score = np.mean(scores['f1-score'])
        print(f'Cross-validation results (n_splits={n_splits}'
              f', n_estimators={n_estimators}, '
              f'max_depth={max_depth}, max_features={max_features}):')
        print(f"Accuracy: {mean_accuracy * 100:.2f}%")
        print(f"Precision: {mean_precision * 100:.2f}%")
        print(f"Recall: {mean_recall * 100:.2f}%")
        print(f"F1-score: {mean_f1_score * 100:.2f}%")

        # Fonction pour entraîner un modèle Random Forest
        def train_rf(X_train, y_train, n_estimators=100, max_depth=None,
                     max_features='sqrt', random_state=42):
            # Initialisation du modèle
            rf = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        random_state=random_state,
                                        min_samples_split=10,
                                        n_jobs=-1)
            # Entraînement du modèle sur les données d'entraînement
            rf.fit(X_train.reshape(-1, X_train.shape[-1]), y_train.ravel())
            return rf

        # Fonction pour tester un modèle Random Forest
        def test_rf(rf, X_test, y_test):
            # Prédiction des étiquettes à partir des données de test
            y_pred = rf.predict(X_test.reshape(-1, X_test.shape[-1]))
            # Calcul des métriques de précision, rappel, F-score et support
            precision, recall, fscore, support = precision_recall_fscore_support(
                y_test.ravel(), y_pred, average='weighted')
            # Affichage des métriques
            print(f"Accuracy: {accuracy_score(y_test.ravel(), y_pred)}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F-score: {fscore}")
            print(f"Support: {support}")

            def cross_validation_rf(input_files, class_names, n_splits=2,
                                    n_estimators=100, max_depth=None,
                                    max_features='sqrt', random_state=42):
                # Création des listes d'entraînement, de test et de validation
                kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
                # Initialisation des scores de validation croisée
                scores = {'accuracy': [], 'precision': [],
                          'recall': [], 'f1-score': []}
                # Calcul de la matrice de confusion moyenne
                conf_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
                for train_index, test_index in kf.split(input_files):
                    # Séparation des données en ensembles d'entraînement et de test
                    X_train, y_train = [], []
                    for file_index in train_index:
                        X, y = read_data(input_files[file_index])
                        X_train.append(X[:, :, 0:15])
                        y_train.append(y)
                    # Fusion des ensembles d'entraînement et de validation
                    X_test, y_test = read_data(input_files[test_index[0]])
                    X_test = X_test[:, :, 0:15]
                    X_train = np.concatenate(X_train)
                    y_train = np.concatenate(y_train)
                    # Entraînement du modèle
                    rf = train_rf(X_train.reshape(-1, X_train.shape[-1]), y_train.ravel(),
                                  n_estimators=n_estimators, max_depth=max_depth,
                                  max_features=max_features, random_state=random_state)
                    # Test du modèle
                    test_rf(rf, X_test.reshape(-1, X_test.shape[-1]), y_test.ravel())
                    # Prédiction des étiquettes pour l'ensemble de test
                    y_pred = rf.predict(X_test.reshape(-1, X_test.shape[-1]))
                    # Calcul de la matrice de confusion
                    conf_matrix += confusion_matrix(
                        y_test.ravel(), y_pred,
                        labels=list(class_names.keys()))
                    accuracy, precision, recall, f1_score = precision_recall_fscore_support(
                        y_test.ravel(), y_pred, labels=list(class_names.values()),
                        average='weighted', zero_division=0)[:4]
                    if (accuracy is not None and
                            precision is not None and
                            recall is not None and
                            f1_score is not None):
                        scores['accuracy'].append(accuracy)
                        scores['precision'].append(precision)
                        scores['recall'].append(recall)
                        scores['f1-score'].append(f1_score)

                    # Calcul de la matrice de confusion moyenne
                conf_matrix = conf_matrix / np.sum(conf_matrix) * 100
                # Affichage graphique de la matrice de confusion moyenne
                plt.figure(figsize=(10, 8))
                sns.set(font_scale=1.4)
                sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.1f',
                            xticklabels=class_names.values(),
                            yticklabels=class_names.values())
                plt.xlabel('Prédiction')
                plt.ylabel('Vérité Terrain')
                plt.title('Matrice de confusion moyenne')
                plt.savefig('visualisation/confusion_matrix.png')
                # Calcul des scores moyens de validation croisée

                mean_scores = {k: np.mean(scores[k]) for k in scores.keys()}
                # Affichage des scores moyens de validation croisée
                mean_accuracy = np.mean(scores['accuracy'])
                mean_precision = np.mean(scores['precision'])
                mean_recall = np.mean(scores['recall'])
                mean_f1_score = np.mean(scores['f1-score'])
                print(f'Cross-validation results (n_splits={n_splits}'
                      f', n_estimators={n_estimators}, '
                      f'max_depth={max_depth}, max_features={max_features}):')
                print(f"Accuracy: {mean_accuracy * 100:.2f}%")
                print(f"Precision: {mean_precision * 100:.2f}%")
                print(f"Recall: {mean_recall * 100:.2f}%")
                print(f"F1-score: {mean_f1_score * 100:.2f}%")


def predict_image(filepath, model, colors):
    # Lecture des données de l'image
    X, y_true = read_data(filepath)
    # Prédiction des étiquettes à partir des données de l'image
    y_pred = model.predict(X.reshape(-1, X.shape[-1]))
    y_pred = y_pred.reshape(y_true.shape)
    # Affichage du rapport de classification
    report = classification_report(y_true.ravel(), y_pred.ravel())
    print(report)
    # Affichage de la précision
    accuracy = accuracy_score(y_true.ravel(), y_pred.ravel())
    print(f"Accuracy: {accuracy}")
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
    # Création de la figure de prédiction
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(rasterio.plot.reshape_as_image(y_pred), cmap='jet')
    plt.axis('off')
    plt.title("Predicted Classes")
    plt.tight_layout()
    # Ajout de la filigrane avec les classes
    classes = list(colors.keys())
    handles = [plt.plot([],[],color=colors[c], ls="", marker=".", \
              markersize=np.sqrt(100), alpha=0.8)[0] for c in classes]
    labels = [f"{c}" for c in classes]
    leg = plt.legend(handles, labels, loc=(1.05, 0.5), fontsize=12, \
               frameon=True, framealpha=1)
    leg.set_title("Classes", prop = {'size':'large'})
    plt.gca().add_artist(leg)
    # Création de la figure de vérité terrain
    fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
    plt.imshow(rasterio.plot.reshape_as_image(y_true), cmap='jet')
    plt.axis('off')
    plt.title("Ground Truth Classes")
    plt.tight_layout()
    # Ajout de la filigrane avec les classes
    handles = [plt.plot([],[],color=colors[c], ls="", marker=".", \
              markersize=np.sqrt(100), alpha=0.8)[0] for c in classes]
    labels = [f"{c}" for c in classes]
    leg = plt.legend(handles, labels, loc=(1.05, 0.5), fontsize=12, \
               frameon=True, framealpha=1)
    leg.set_title("Classes", prop = {'size':'large'})
    plt.gca().add_artist(leg)
    # Enregistrement des figures
    name = os.path.splitext(os.path.basename(filepath))[0]
    plt.savefig(f"visualisation/{name}_prediction.png", dpi=300)
    plt.savefig(f"visualisation/{name}_ground_truth.png", dpi=300)
    # Affichage de la matrice de confusion
    print("Confusion Matrix")
    print(cm)
    return report, accuracy, cm, fig, fig_gt