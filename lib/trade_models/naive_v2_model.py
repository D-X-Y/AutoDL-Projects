##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021 #
##################################################
# A Simple Model that reused the prices of last day
##################################################
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import pandas as pd

from qlib.log import get_module_logger

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class NAIVE_V2(Model):
    """NAIVE Version 2 Quant Model"""

    def __init__(self, d_feat=6, seed=None, **kwargs):
        # Set logger.
        self.logger = get_module_logger("NAIVE")
        self.logger.info("NAIVE version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.seed = seed

        self.logger.info(
            "NAIVE parameters setting: d_feat={:}, seed={:}".format(
                self.d_feat, self.seed
            )
        )

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.fitted = False

    def process_data(self, features):
        features = features.reshape(len(features), self.d_feat, -1)
        features = features.transpose((0, 2, 1))
        return features[:, :59, 0]

    def mse(self, preds, labels):
        masks = ~np.isnan(labels)
        masked_preds = preds[masks]
        masked_labels = labels[masks]
        return np.square(masked_preds - masked_labels).mean()

    def model(self, x):
        x = 1 / x - 1
        masks = ~np.isnan(x)
        results = []
        for rowd, rowm in zip(x, masks):
            temp = rowd[rowm]
            if rowm.any():
                results.append(float(rowd[rowm][-1]))
            else:
                results.append(0)
        return np.array(results, dtype=x.dtype)

    def fit(self, dataset: DatasetH):
        def _prepare_dataset(df_data):
            features = df_data["feature"].values
            features = self.process_data(features)
            labels = df_data["label"].values.squeeze()
            return dict(features=features, labels=labels)

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        train_dataset, valid_dataset, test_dataset = (
            _prepare_dataset(df_train),
            _prepare_dataset(df_valid),
            _prepare_dataset(df_test),
        )
        # df_train['feature']['CLOSE1'].values
        # train_dataset['features'][:, -1]
        train_mse_loss = self.mse(
            self.model(train_dataset["features"]), train_dataset["labels"]
        )
        valid_mse_loss = self.mse(
            self.model(valid_dataset["features"]), valid_dataset["labels"]
        )
        self.logger.info("Training MSE loss: {:}".format(train_mse_loss))
        self.logger.info("Validation MSE loss: {:}".format(valid_mse_loss))
        self.fitted = True

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("The model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature")
        index = x_test.index

        preds = self.model(self.process_data(x_test.values))
        return pd.Series(preds, index=index)
