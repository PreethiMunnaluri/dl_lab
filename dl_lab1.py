import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names

Data Pre-processing: Standardization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Perform PCA for Dimension Reduction

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

Scree Plot

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1),
         explained_variance,
         marker='o',
         linestyle='--')

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid()
plt.show()

Data Visualization in Lower Dimension (2D)

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=X_pca_2,
    columns=['PC1', 'PC2']
)

pca_df['Target'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Target',
    palette='Set1',
    data=pca_df
)

plt.title('PCA – 2D Visualization of Iris Dataset')
plt.show()