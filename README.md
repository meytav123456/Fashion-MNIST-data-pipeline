# Fashion-MNIST Dimensionality Reduction & Clustering 

This repository contains an end-to-end unsupervised learning pipeline that compares classical and deep-learning approaches for dimensionality reduction, followed by a comprehensive clustering analysis on the Fashion-MNIST dataset.

##  Project Overview
The main objective is to reduce the dimensionality of 28x28 images into a 32-dimensional latent space and evaluate which reduction method yields better, more separable clusters without using ground-truth labels.

### Key Comparisons:
1. **Dimensionality Reduction:** Truncated SVD (Linear) vs. Convolutional Autoencoder (Non-linear).
2. **Clustering Algorithms:** K-Means vs. Gaussian Mixture Models (GMM) vs. DBSCAN.



---

##  Methodology

### 1. Data Preprocessing
* Filtered the Fashion-MNIST dataset to a subset of 7 selected classes.
* Data was normalized and mean-centered before applying linear dimensionality reduction.

### 2. Dimensionality Reduction Techniques
* **Truncated SVD:** Applied to the centered data to extract 32 principal components. A 2D projection was also created for baseline visualization.
* **Convolutional Autoencoder (CAE):** Built with PyTorch using `Conv2d` and `ConvTranspose2d` layers to compress the spatial data into a 32D latent vector, trained using MSE Loss. Latent path interpolations were generated to verify the model learned a continuous representation.

### 3. Clustering Suite
The 32D features extracted from both SVD and the Autoencoder were clustered using:
* **K-Means** (Partitioning-based)
* **GMM** (Distribution-based / Expectation-Maximization)
* **DBSCAN** (Density-based, utilizing K-Nearest Neighbors to find optimal epsilon)

---

##  Final Results & Evaluation
The clustering performance of both reduction methods (SVD vs. AE) was evaluated using internal and external metrics.

| Dimension Reduction | Clusterization | ACC | ARI | NMI | Silhouette |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVD** | K-Means | [ACC] | [ARI] | [NMI] | [Sil] |
| **SVD** | GMM | [ACC] | [ARI] | [NMI] | [Sil] |
| **SVD** | DBSCAN | [ACC] | [ARI] | [NMI] | [Sil] |
| **AE** | K-Means | [ACC] | [ARI] | [NMI] | [Sil] |
| **AE** | GMM | [ACC] | [ARI] | [NMI] | [Sil] |
| **AE** | DBSCAN | [ACC] | [ARI] | [NMI] | [Sil] |

> **Note:** Fill in the values from the final output dataframe.

---

##  Visualizations
The project includes rich visual analysis:
* Autoencoder training loss progression.
* Original vs. Reconstructed images.
* 2D Latent space scatter plots (projected via SVD) showing true labels vs. predicted clusters.
* Latent path interpolation morphing one clothing item into another.



## 🛠️ Requirements
* Python 3.x
* PyTorch & Torchvision
* Scikit-Learn
* SciPy
* Pandas & NumPy
* Matplotlib
