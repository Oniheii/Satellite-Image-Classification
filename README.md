# Satellite Image Classification with Random Forest and Bagging

This project focuses on the pixel-level classification of satellite images using advanced machine learning techniques. Specifically, we implement a **Random Forest** model and improve its performance by incorporating **Bootstrap Aggregating (Bagging)** for remote sensing applications. The data used is from the **Sentinel-2** satellite, part of the **Copernicus program** by the European Space Agency.

### Objective
The goal of this project is to classify land cover types such as grassland, broadleaf vegetation, coniferous vegetation, built-up areas, water bodies, and croplands based on satellite imagery from the Berlin region. We aim to enhance the accuracy of classification through the **Bagging** method applied to the **Random Forest** model.

### Data
The raw data consists of satellite images downloaded from the **Copernicus Open Access Hub**. Images were captured between **January 2019 and December 2019**, with cloud coverage below 10%. The images were preprocessed to focus on a 60m resolution, and multiple spectral bands (e.g., red, green, infrared) were merged into **GeoTIFF** format for further analysis.

The classification process relies on spectral information derived from **NDVI** (Normalized Difference Vegetation Index) and shortwave infrared bands (**SWIR**), which help in identifying water bodies and vegetation types.

### Methodology
- **Random Forest**: Used for pixel-wise classification based on spectral information from satellite images.
- **Bagging**: Applied to reduce overfitting and enhance the model's generalization.
- **GeoTIFF Creation**: Preprocessing step where multi-band JP2 images are merged into single GeoTIFF files for each region.
- **Labeling**: Pixels are classified into six distinct classes: Grassland, Broadleaf Vegetation, Coniferous Vegetation, Built-up areas, Water, and Cropland.

### Results
While the Random Forest model achieved over **90% accuracy** in pixel classification, the final step of visualizing the classified map was not completed within the project timeline. Future work will focus on completing this visualization and comparing the results with other classification methods.

### Next Steps
- Finalize the mapping visualization to present the classification results.
- Compare the performance of Random Forest and Bagging with other statistical models in remote sensing.
  
### How to Use
To run the code, download the required raw satellite data from the following link:  
[Sentinel-2 Data](https://drive.google.com/drive/folders/1dtIIx3MO7OBzFTM9tDMQX0P4SIr4SnmF?usp=share_link)

### File Structure
- `preprocessing.py`: Script for downloading and preprocessing the raw satellite data.
- `classification_model.py`: Implementation of the Random Forest model with Bagging for satellite image classification.
- `evaluation.py`: Code for evaluating model performance and accuracy.
- `visualization.py`: (To be completed) Script for visualizing classified satellite maps.
