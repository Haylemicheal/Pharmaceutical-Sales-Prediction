#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging as log


log.basicConfig(filename="../logs/exploration.txt", format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
class Exploration:
    """
    A module for data visualization and exploration
    """
    def get_missing(self, df):
        """
        A method for getting missing values
        Args:
            df: dataframe
        Return:
            %missing: Percent of missing values
            missingCount: The number of missing values
        """
        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMissing = missingCount.sum()
        log.info("Get the missing values")
        return round((totalMissing/totalCells), 2) * 100, missingCount