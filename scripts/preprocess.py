import os
import pandas as pd
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import logging as log

log.basicConfig(filename="../logs/preprocess.txt", format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
class Preprocess:
    """
    A class for data preprocessing
    """
    def get_numerical_columns(self, df: pd.DataFrame) -> list:
        """
        A method to get numerical columns
        Args:
            df: Pandas dataframe
        Returns:
            Numerical columns
        """
        log.info("Get the numerical columns")
        return df.select_dtypes(include=['number']).columns.to_list()

    def get_categorical_columns(self, df: pd.DataFrame) -> list:
        """
        A method to get categorical columns
        Args:
            df: Pandas dataframe
        Returns:
            Categorical cols
        """
        log.info("Get the categorical columns")
        return df.select_dtypes(exclude=['number']).columns.to_list()

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode features using LabelEncoder.
        Args:
            df: Pandas dataframe
        Return:
            df: encoded dataframe
        """
        features = self.get_categorical_columns(df)
        for feature in features:
            le = LabelEncoder()
            le.fit(df[feature])
            df[feature] = le.transform(df[feature])
        log.info("Encode the features")
        return df
    
    def normalizer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical columns
        Args:
            df: Pandas dataframe
        Returns:
            normalized dataframe
        """
        norm = Normalizer()
        log.info("Normalize the data")
        return pd.DataFrame(norm.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))

    def min_max_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        Args:
            df: Pandas dataframe
        Returns:
            MinMaxScaled dataframe
        """
        minmax_scaler = MinMaxScaler()
        log.info("Do min max scale the data")
        return pd.DataFrame(minmax_scaler.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))

    def standard_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        Args:
            df: Pandas dataframe
        Returns:
            Scaled dataframe
        """
        standard_scaler = StandardScaler()
        log.info("Standard scale the data")
        return pd.DataFrame(standard_scaler.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))

    def prepare_data(self, data):
        """
        Prepare data for training using skleaern pipline
        Args:
            data: Training dataset
        Return:
            preprocessed data
        """
        preprocess_pipeline = Pipeline(steps=[
            ('set_type', FunctionTransformer(self.set_type, validate=False)),
            ('label_encoder', FunctionTransformer(
                self.encode_features, validate=False)),
            ('scaler', FunctionTransformer(self.standard_scaler, validate=False)),
        ])
        log.info("Data preparation")
        return preprocess_pipeline.fit_transform(data)


    def set_type(self, data):
        """
        A method to set a datatype
        Args:
            data: Training data
        Return:
            data: A data with fixed datatype
        """
        try:
            data['Store'] = data['Store'].astype('int')
            data['DayOfWeek'] = data['DayOfWeek'].astype('int')
            data['Date'] = pd.to_datetime(data['Date'])
            data['Customers'] = data['Customers'].astype('int')
            data['Open'] = data['Open'].astype('int')
            data['Promo'] = data['Promo'].astype('int')
            data['StateHoliday'] = data['StateHoliday'].astype('object')
            data['SchoolHoliday'] = data['SchoolHoliday'].astype('int')
            data['StoreType'] = data['StoreType'].astype('object')
            data['Assortment'] = data['Assortment'].astype('object')
            data['CompetitionDistance'] = data['CompetitionDistance'].astype('float')
            data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].astype('float')
            data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].astype('float')
            data['Promo2'] = data['Promo2'].astype('int')
            data['Promo2SinceWeek'] = data['Promo2SinceWeek'].astype('float')
            data['Promo2SinceYear'] = data['Promo2SinceYear'].astype('float')
            data['PromoInterval'] = data['PromoInterval'].astype('object')
            log.info("Set type of each column")
        except:
            log.error("Unable to set data type")

        return data