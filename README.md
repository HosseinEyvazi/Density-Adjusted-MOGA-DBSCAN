# Enhanced MOGA-DBSCAN

This repository contains the implementation of **Enhanced MOGA-DBSCAN**, a novel algorithm that improves the clustering quality of DBSCAN by optimizing its parameters using a Multi-Objective Genetic Algorithm (MOGA). The algorithm also introduces a Density-Adjusted Outlier Index for better evaluation of outliers, and parallelizes its computation for scalability.

For more details, read the full paper presented at the **First International Conference on Machine Learning and Knowledge Discovery (MLKD 2024)**:
[Enhanced MOGA-DBSCAN: Improving Clustering Quality with a Density-Adjusted Outlier Index](https://mlkd.aut.ac.ir/MLKD-2024-Abstracts/19.pdf)

## Key Features

- **Optimized DBSCAN Parameters**: Utilizes a Multi-Objective Genetic Algorithm (MOGA) to optimize `Eps` and `MinPts` for better clustering performance.
- **Density-Adjusted Outlier Index**: Enhances the accuracy of outlier detection by considering cluster density.
- **Parallelized Computation**: Reduces runtime, making the algorithm applicable to larger datasets.
- **Improved Clustering Metrics**: Achieves higher Silhouette scores and Rand indices compared to traditional approaches.

## Abstract

> Clustering is a crucial aspect of data mining and machine learning, and its performance can significantly depend on parameter selection. The DBSCAN algorithm, known for its efficacy in detecting clusters of arbitrary shapes, relies heavily on its two parameters: `Eps` and `MinPts`. This paper presents an enhanced version of the Multi-Objective Genetic Algorithm (MOGA) for optimizing the parameters of DBSCAN, named Enhanced MOGA-DBSCAN. Our approach incorporates a modified Outlier Index that accounts for the density of clusters, providing a better evaluation of outliers. Additionally, we parallelized the computation of the Outlier Index to significantly reduce the runtime, enabling practical applicability to larger datasets. Experimental results on two benchmark datasets demonstrate that Enhanced MOGA-DBSCAN outperforms the original MOGA-DBSCAN algorithm, achieving higher Silhouette scores and Rand indices while requiring less computational time. This advancement not only improves clustering efficiency but also offers more meaningful insights into the underlying data structure.

## Installation

### Prerequisites

- Python 3.8+
- Required Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`

Install the required packages using:
```bash
pip install -r requirements.txt
