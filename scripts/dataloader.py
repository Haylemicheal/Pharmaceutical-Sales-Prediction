#!/usr/bin/env python3
import logging
import pandas as pd

logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')


class DataLoader:
    """A class for data loading
    """
    def __init__(self):
        logging.info("The data loader instance is created")
    
    def read_csv(self, path):
        """Read csv file
           Args:
                path: location of the csv file
           Return:
                df: pandas dataframe
        """
        df = pd.read_csv(path, sep=",", low_memory=False)
        return df
