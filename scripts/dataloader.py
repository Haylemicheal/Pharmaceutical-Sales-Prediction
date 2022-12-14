#!/usr/bin/env python3
import logging as log
import pandas as pd
import sys, os, io
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Model

log.basicConfig(filename='../logs/dataloader.txt', format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')


class DataLoader:
    """A class for data loading
    """
    def __init__(self):
        log.info("The data loader instance is created")
    
    def read_csv(self, path):
        """Read csv file
           Args:
                path: location of the csv file
           Return:
                df: pandas dataframe
        """
        df = pd.read_csv(path, sep=",", low_memory=False)
        log.info("Read csv file")
        return df
