path=  'C:/Users/felipe/Documents/Brain/'
import sys
sys.path.append('C:/Users/felipe/Documents/Brain/') 

from sklearn.ensemble import RandomForestRegressor
from base_regressor_ import BaseRegressor
from skopt.space import Real, Categorical, Integer

class RFRegressor(BaseRegressor):
    def __init__(self, save_path=None, scaler=None, params=None, params_space=None, fit_params_search=None,model_params_search=None,fit_params_train=None,model_params_train=None, name_model="MLP"):
        super().__init__(save_path, scaler, params, params_space, fit_params_search ,model_params_search,fit_params_train ,model_params_train, name_model)
        
        self.model_ml= RandomForestRegressor
        self.params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,            
            'bootstrap': True
        }
        
        self.params_space = {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 100),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical([ 'sqrt', 'log2']),
            'bootstrap': [True, False] 
        }
