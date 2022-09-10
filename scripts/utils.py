import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class Util:
    """
    A class that contains helper functions
    """
    def save_model(self, model):
        """
        A method to save the model
        Args:
            model: A model to be saved
        """
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
        pickle.dump(model, open(f'../models/{timestamp}.pkl', 'wb'))
    def feature_importance_plot(self, data):
        """
        Plot of features
        Args:
            data: Data
        """
        sns.barplot(data=feat_imp,  x="importance", y="features")
        plt.title("Feature importance using random forest", size=18)
        plt.xticks(rotation=60, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("importance", fontsize=12)
        plt.ylabel("features", fontsize=12)
        plt.show()
