# Unsupervised Learning on Fashion-MNIST: Dimensionality Reduction & Clustering

## Overview
This project investigates the intersection of **Dimensionality Reduction** and **Unsupervised Clustering** using the Fashion-MNIST dataset. We compare the efficacy of linear reduction (Truncated SVD) against non-linear representations learned by a Convolutional Autoencoder (ConvAE). By compressing $28 \times 28$ high-dimensional image data into constrained latent spaces ($d=2$ and $d=32$), we evaluate how different bottleneck topologies affect the separability of classes when subjected to K-Means, Gaussian Mixture Models (GMM), and DBSCAN.

---

## Dataset & Preprocessing
The model is trained and evaluated on a subset of the Fashion-MNIST dataset, focusing on 7 visually challenging classes: *T-shirt/top, Trouser, Dress, Shirt, Sneaker, Bag, and Ankle boot*. 

To prepare the data for both PyTorch tensors and Scikit-Learn algorithms, we applied:
* **Mean Centering:** Subtracting the training mean vector from all samples to align the data around the origin, which is mathematically critical for SVD.
* **Standardization:** Using `StandardScaler` on the latent features prior to clustering to ensure isotropic feature scaling.

---

## Dimensionality Reduction Architectures

### 1. Linear Reduction: Truncated SVD
We established a baseline using Truncated Singular Value Decomposition (SVD), projecting the centered images into $k=32$ principal components.

### 2. Non-Linear Reduction: Convolutional Autoencoder (ConvAE)
We designed a Convolutional Autoencoder in PyTorch to capture non-linear spatial hierarchies. 
* **Encoder:** Two symmetric `Conv2d` layers with ReLU activations, compressing the $28 \times 28$ input into $7 \times 7$ feature maps before flattening into the latent dimension.
* **Decoder:** `ConvTranspose2d` layers reconstructing the original resolution. 
* **Loss Function:** Minimized using Mean Squared Error (MSE) with `reduction="sum"`. 

To observe the "bottleneck limit," we trained two separate variants: $d=2$ and $d=32$.

<img width="910" height="588" alt="image" src="https://github.com/user-attachments/assets/d92114f7-460f-4f34-811e-ccd85aad6feb" />

<img width="939" height="597" alt="image" src="https://github.com/user-attachments/assets/5707f0f1-2f9f-48e4-b097-cb521c72fb23" />

> Training and testing MSE loss over epochs, demonstrating the superior reconstruction capacity of the $d=32$ and the $d=2$ architectures.*

---

## Latent Space Analysis & Reconstruction

### Image Reconstruction
Extreme compression to $d=2$ results in severe information loss, producing blurred generic silhouettes. Conversely, the $d=32$ model preserves high-frequency details necessary to distinguish between topologically similar classes (e.g., Shirts vs. T-shirts).


<img width="850" height="364" alt="image" src="https://github.com/user-attachments/assets/5400ddec-8003-4262-9ed3-3abb365be797" />
<img width="846" height="383" alt="image" src="https://github.com/user-attachments/assets/6d5e23fc-4d9a-4328-8ab0-518eff98d195" />


> * Qualitative comparison of original test images against $d=2$ and $d=32$ ConvAE reconstructions.*

### Latent Path Interpolation
To verify that the Autoencoder learned a continuous, meaningful manifold rather than simply memorizing data points, we performed linear interpolation between the latent vectors of two distinct classes. The smooth visual transition in the decoded path proves the structural integrity of the latent space.

![Latent Interpolation] <img width="1700" height="436" alt="image" src="https://github.com/user-attachments/assets/98df901e-1254-4a27-81bd-4cf073e626af" />

> *Figure 3: Linear interpolation in the $d=32$ latent space between Class 0 (T-shirt) and Class 1 (Trouser).*

### Latent Space Visualization
Projecting the 32-dimensional embeddings into a 2D plane reveals a dense, continuous manifold. The lack of strict convexity explains the varying success rates of different clustering algorithms.

<img width="798" height="563" alt="image" src="https://github.com/user-attachments/assets/64cd8bcd-c275-4b40-87e7-b7ab4309c809" />
<img width="811" height="624" alt="image" src="https://github.com/user-attachments/assets/6e284ab8-22c8-484d-bd85-3b43c4ca44d6" />


> *Figure 4: The ConvAE latent space distribution. Note the significant density bridging between visually similar upper-body garments.*

---

## Clustering Algorithms & Tuning
We applied unsupervised clustering to the $d=32$ representations (both SVD and ConvAE) and evaluated them using the **Hungarian Algorithm** for optimal bipartite label matching (ACC), Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Silhouette Score.

1. **K-Means:** Assumes spherical, isotropic clusters. It performs moderately well but struggles in the overlapping, non-convex regions of the clothing manifold.
2. **Gaussian Mixture Models (GMM):** Utilizes Expectation-Maximization to fit full covariance matrices. It consistently outperformed K-Means by accommodating the elliptical variance of the data "clouds."
3. **DBSCAN (Density-Based):** * *The Challenge:* Fashion-MNIST embeddings suffer from **Density Bridging**—there are no clear "empty" boundaries between similar classes. 
   * *Tuning:* We implemented a custom geometrical knee-point detection on a K-Nearest Neighbors distance array, combined with a high-resolution grid search ($\Delta \epsilon = 0.01$), to identify the optimal $\epsilon$ parameter and force cluster separation.

<img width="1706" height="583" alt="image" src="https://github.com/user-attachments/assets/20db7654-39c1-490b-96ee-ac9ec2308f4d" />

> *Figure 5: Visual comparison of K-Means, GMM, and DBSCAN assignments.*

---

## 🏆 Final Results & Metrics

*(Below are the empirical metrics derived from evaluating the $d=32$ models on the test set)*

| Dimension Reduction | Clustering Algorithm | ARI | NMI | Silhouette | ACC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVD** | KMeans | `[0.4376]` | `[0.5514]` | `[0.241]` | `[0.6126]` |
| **SVD** | GMM | `[	0.4748]` | `[0.605]` | `[0.1696]` | `[0.6429]` |
| **SVD** | DBSCAN | `[0.0768]` | `[0.1312]` | `[-0.0805]` | `[0.2567]` |
| **AE ($d=32$)** | KMeans | `[0.5879]` | `[0.6711]` | `[0.1964]` | `[0.7677]` |
| **AE ($d=32$)** | GMM | `[0.6499]` | `[0.7093]` | `[0.1641]` | `[0.8194]` |
| **AE ($d=32$)** | DBSCAN | `[0.2035]` | `[0.4114]` | `[-0.0065]` | `[0.413]` |

### Key Conclusions
1. **Representational Power:** Non-linear features extracted via ConvAE yield structurally superior representations for probabilistic clustering (GMM) compared to linear SVD.
2. **Topological Limitations:** While dimensionality expansion from 2 to 32 provides mathematical room for separation, the inherent continuity of fashion image data severely limits the effectiveness of density-based clustering like DBSCAN.
