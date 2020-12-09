import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.datasets import load_breast_cancer, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


def main():
    # gerando dados sintéticos com 4 classes
    X, y = make_blobs(n_samples=10000, n_features=2, centers = 8, cluster_std=.8,random_state=12345)
    y = y%4
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.grid(True)

if __name__ == "__main__":
    main()
