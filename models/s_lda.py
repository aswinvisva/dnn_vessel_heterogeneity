import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

import models.spatial_lda.model as s_lda
from models.spatial_lda.featurization import featurize_samples, neighborhood_to_avg_marker, neighborhood_to_cluster, \
    neighborhood_to_count, make_merged_difference_matrices

"""
Using Spatial LDA on our dataset, originally written by: Zhenghao Chen, Ilya Soifer, Hugo Hilton, Leeat Keren, and 
Vladimir Jojic
"""


class SpatialLDA:

    def __init__(self, n_topics, x_labels):
        self._n_topics = n_topics
        self._model = None
        self._x_labels = x_labels
        self.x_col = "X"
        self.y_col = "Y"
        self.is_anchor_col = "IsAnchor"

    def fit_predict(self, all_samples_marker_data, all_samples_coordinates):
        x = self._generate_features(all_samples_marker_data)
        diff_matrices = self._compute_difference_matrices(x, all_samples_coordinates)

        self._model = s_lda.train(x, diff_matrices, self._n_topics, verbosity=2)

        return self.model.fit_transform(x)

    def _compute_difference_matrices(self, sample_features, all_points_coordinates):
        coord_dfs = {}

        for i, coords in enumerate(all_points_coordinates):
            df = pd.DataFrame(coords, columns=["x", "y"])
            coord_dfs[i] = df

        difference_matrices = make_merged_difference_matrices(sample_features, coord_dfs)

        return difference_matrices

    def _generate_features(self, all_samples_marker_data):
        """
        Generate sample dataframes as done in the Spatial LDA paper
        :param all_samples_marker_data:
        :return:
        """
        sample_dfs = []

        column_names = []
        column_names.extend(self._x_labels)

        for i in range(len(all_samples_marker_data)):
            samples = all_samples_marker_data[i]
            all_rows = []

            for vessel_idx, data in enumerate(samples):
                all_rows.append(data.tolist())

            df = pd.DataFrame(np.array(all_rows), columns=column_names)
            df.index = map(lambda x: (i, x), df.index)

            sample_dfs.append(df)

        all_sample_features = pd.concat(sample_dfs).fillna(0)

        return all_sample_features
