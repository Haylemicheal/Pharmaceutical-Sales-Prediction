from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging as log
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from scripts.preprocess import Preprocess

log.basicConfig(filename="../logs/modeling.txt", format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
class Model:
    """
    Class for machine learning modeling
    """
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.data_process = Preprocess()

    def preprocess(self):
        """Preprocess Pipline"""
        pipeline = Pipeline(steps=[
            ('label_encoder', FunctionTransformer(
                self.data_process.encode_features, validate=False)),
            ('scaler', FunctionTransformer(self.data_process.standard_scaler, validate=False)),
            ('target_feature_split', FunctionTransformer(self.split_target_feature, kw_args={'target_col':'Sales'}, validate=False)),
            ('train_test_split', FunctionTransformer(self.split_train_test_val,kw_args={'size':(.7,.2,.1)}, validate=False))
        ])
    
        log.info("Preprocess the training data for modeling")
        return pipeline.fit_transform(self.data)

    def split_target_feature(self, df: pd.DataFrame, target_col: str) -> tuple:
        """
        Split the target Column
            Args:
                df: dataset
                target_col: The target column
            return:
                features: The dataframe except the target
                target: The target column
        """
        target = df[[target_col]]
        features = df.drop(target_col, axis=1)
        log.info("Split the target and the feature")
        return features, target

    def split_train_test_val(self, input_data:tuple, size:tuple)-> list:
        """
        Split the data into train, test and validation.
        """
        X,Y = input_data
        train_x, temp_x, train_y, temp_y = train_test_split(X, Y, train_size=size[0], test_size=size[1]+size[2], random_state=42)
        test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y, train_size=size[1]/(size[1]+size[2]), test_size=size[2]/(size[1]+size[2]), random_state=42)
        log.info("Split the data to training, test, and validation")
        return [train_x, train_y, test_x, test_y, val_x, val_y]

    def eval_metrics(self,actual, pred):
        """
        Evaluation metric
        Args:
            actual: The actual value
            pred: That predicted vale
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        log.info("Do evaluation using rmse, mae, and r2")
        return rmse, mae, r2

    def train(self, args, exp_name):
        """
        Training the model
        Args:
            args: arguments
            exp_name: Experiment name
        """
        mlflow.set_experiment(exp_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run():
            self.preprocessed_data = self.preprocess()
            model = self.model(**args)
            fitted_model = model.fit(self.preprocessed_data[0], self.preprocessed_data[1].values.ravel())
            y_pred = fitted_model.predict(self.preprocessed_data[2])
            (rmse, mae, r2) = self.eval_metrics(self.preprocessed_data[3], y_pred)
            mlflow.log_param("n_estimators", args['n_estimators'])
            mlflow.log_param("max_features", args['max_features'])
            mlflow.log_param("max_depth", args['max_depth'])
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            result_df = self.preprocessed_data[2].copy()
            result_df["Prediction Sales"] = y_pred
            result_df["Actual Sales"] = self.preprocessed_data[3]
            result_agg = result_df.groupby("day").agg(
            {"Prediction Sales": "mean", "Actual Sales": "mean"})

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(fitted_model, "model", registered_model_name="RandomForestModel")
            else:
                mlflow.sklearn.log_model(fitted_model, "model") 
            log.info("Training the model")
            
        return fitted_model, result_agg

    def get_features_importance(self,fitted_model):
        """
        Feature importance
        """
        importance = fitted_model.feature_importances_
        f_df = pd.DataFrame(columns=["features", "importance"])
        f_df["features"] = self.preprocessed_data[0].columns.to_list()
        f_df["importance"] = importance
        return f_df

    def prediction_graph(self, res_dataframe):
        """
        The prediction graph
        Args:
            res_dataframe: input dataframe
        """
        fig = plt.figure(figsize=(18, 5))
        sns.lineplot(x=res_dataframe.index,
                     y=res_dataframe["Actual Sales"], label='Actual')
        sns.lineplot(x=res_dataframe.index,
                     y=res_dataframe["Prediction Sales"], label='Prediction')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel="Day", fontsize=16)
        plt.ylabel(ylabel="Sales", fontsize=16)
        plt.show()
    

    
