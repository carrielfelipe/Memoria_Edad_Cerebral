import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import shap
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
from joblib import dump, load
from nilearn import plotting
import statsmodels.api as sm
from sklearn.model_selection import StratifiedShuffleSplit


class BaseRegressor:
    def __init__(self,save_path=None, scaler=None, params=None, params_space=None, fit_params_search=None, model_params_search=None,fit_params_train=None, models_params_train=None, name_model=None):             
        
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.params = params if params is not None else {}
        self.params_space = params_space if params_space is not None else {}
        self.fit_params_search = fit_params_search if fit_params_search is not None else {}
        self.model_params_search = model_params_search if model_params_search is not None else {}
        self.fit_params_train = fit_params_train if fit_params_train is not None else {}
        self.model_params_train = models_params_train if models_params_train is not None else {}
        self.save_path = save_path
        self.model_ml = None
        self.name_model = name_model
        self.model = None
        self.opt_model = None
        self.explainer = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.residual_model = None
        
        
    def preprocess_data(self, X):
        """
        Escala las características utilizando MinMaxScaler.
        """
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def set_data(self,X,y, preprocess=True):
        if preprocess:
            X_scaled = self.preprocess_data(X)
        else:
            X_scaled = X
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
    
    def set_data_2(self, X, y):
        X_scaled = self.preprocess_data(X)
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
        for train_index, test_index in strat_split.split(X, y):
            self.X_train, self.X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
        
    def search_best_model(self,  X=None, y=None, param_space_=None, n_iter_=10, n_jobs_=-1,kf_mode=1,nbins=10):
        """
        Realiza la búsqueda de hiperparámetros utilizando BayesSearchCV.
        """
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        if param_space_ is None:
            param_space = self.params_space
        else:
            param_space = param_space_

        if kf_mode==1:
            n_splits = 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=126)

        if kf_mode==2:
            n_bins = nbins  # Número de bins, puedes ajustarlo según la distribución de tus datos
            y_binned = np.digitize(y, bins=np.linspace(min(y), max(y), n_bins))
            y=y_binned
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=126)

       
        
        self.opt_model = BayesSearchCV(
            estimator=self.model_ml(**self.model_params_search),
            search_spaces=param_space,
            #fit_params=self.fit_param,
            cv=kf,
            n_iter=n_iter_,
            scoring='neg_mean_absolute_error',
            n_jobs=n_jobs_,
            random_state=42,
            verbose=1
        )
                
        self.opt_model.fit(X, y, **self.fit_params_search)
        best_params_return = dict(self.opt_model.best_params_)  

        return self.opt_model, best_params_return
    
    def trainer(self, X_train=None,X_test=None, y_train=None,y_test=None,  params_=None, kf=None):
        """
        Entrena un modelo de regresión XGBoost y evalúa su desempeño utilizando validación cruzada.
        """   
        if X_train is None:
            X_train = self.X_train
        if X_test is None:
            X_test = self.X_test
        if y_train is None:
            y_train = self.y_train
        if y_test is None:
            y_test = self.y_test

        if params_ is None:
            params = self.params
        else:
            params = params_

    
        best_fold = 0
        best_score = float('inf')        
        best_model = None
        

        metrics = ['mae', 'mse', 'rmse', 'r2']
        results = {'train': {m: [] for m in metrics}, 
                'val': {m: [] for m in metrics}, 
                'test': {m: [] for m in metrics},
                'model': []}
        
        if kf is None:
            n_splits = 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=126)

        for fold, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
            X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[test_index]

            model = self.model_ml(**params,**self.model_params_train)
            model.fit(X_train_kf, y_train_kf,**self.fit_params_train)
                    

            y_pred_train = model.predict(X_train_kf)
            train_metrics = self.regression_metrics(y_train_kf, y_pred_train)

            y_pred_val = model.predict(X_val_kf)
            val_metrics = self.regression_metrics(y_val_kf, y_pred_val)

            y_pred_test = model.predict(X_test)
            test_metrics = self.regression_metrics(y_test, y_pred_test)

            # Almacenar los resultados de las métricas
            for ds in ['train', 'val', 'test']:
                if ds == 'train':
                    metrics_set = train_metrics
                elif ds == 'val':
                    metrics_set = val_metrics
                else:
                    metrics_set = test_metrics
                    
                for i, metric in enumerate(metrics):
                    results[ds][metric].append(metrics_set[i])
            
            # Almacenar el modelo
            results['model'].append(model)
            
            mae = val_metrics[0]
            if mae < best_score:
                best_fold = fold
                best_score = mae                
                best_model = model
                      
        self.model=best_model

        best_model_results_ = (test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3])
        best_model_results = (best_model, best_fold) + best_model_results_
        
        return results, best_model_results, best_model       
        
    def predicter(self, X_test=None):
        if X_test is None:
            X_test = self.X_test
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def predicter_corrected(self, X_test=None, y_test=None, X_train=None):
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        if X_train is None:
            X_train = self.X_train
        y_pred_train=self.predicter(X_train)
        y_pred_test=self.predicter(X_test)

        train_residuals = y_pred_train - self.y_train.to_numpy()
        residual_model = LinearRegression()
        residual_model.fit(self.y_train.to_numpy().reshape(-1, 1), train_residuals)
        correction = residual_model.predict(y_test.to_numpy().reshape(-1, 1))
        y_pred_adjusted = y_pred_test - correction
        
        return y_pred_adjusted
    
    def trainer_2(self, X_train=None,X_test=None, y_train=None,y_test=None,  params_=None, save_result=False, kf=None):
        """
        Entrena un modelo de regresión XGBoost y evalúa su desempeño utilizando validación cruzada.
        """   
        if X_train is None:
            X_train = self.X_train
        if X_test is None:
            X_test = self.X_test
        if y_train is None:
            y_train = self.y_train
        if y_test is None:
            y_test = self.y_test

        if params_ is None:
            params = self.params
        else:
            params = params_

    
        best_fold = 0
        best_score = float('inf')        
        best_model = None
        

        metrics = ['mae', 'mse', 'rmse', 'r2']
        results = {'train': {m: [] for m in metrics}, 
                'val': {m: [] for m in metrics}, 
                'test': {m: [] for m in metrics},
                'model': []}
        
        if kf is None:
            n_splits = 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=126)

        for fold, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
            X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[test_index]

            model = self.model_ml(**params,**self.model_params_train)
            model.fit(X_train_kf, y_train_kf,**self.fit_params_train)
                    

            y_pred_train = model.predict(X_train_kf)
            y_pred_val = model.predict(X_val_kf)
            y_pred_test = model.predict(X_test)

            train_residuals = y_pred_train - y_train_kf.to_numpy()
            residual_model = LinearRegression()
            residual_model.fit(y_train_kf.to_numpy().reshape(-1, 1), train_residuals)
            correction_val = residual_model.predict(y_val_kf.to_numpy().reshape(-1, 1))
            correction_test = residual_model.predict(y_test.to_numpy().reshape(-1, 1))

            y_pred_adjusted_val = y_pred_val - correction_val
            y_pred_adjusted_test = y_pred_test - correction_test


            train_metrics = self.regression_metrics(y_train_kf, y_pred_train)
            val_metrics = self.regression_metrics(y_val_kf, y_pred_adjusted_val)
            test_metrics = self.regression_metrics(y_test, y_pred_adjusted_test)

            # Almacenar los resultados de las métricas
            for ds in ['train', 'val', 'test']:
                if ds == 'train':
                    metrics_set = train_metrics
                elif ds == 'val':
                    metrics_set = val_metrics
                else:
                    metrics_set = test_metrics
                    
                for i, metric in enumerate(metrics):
                    results[ds][metric].append(metrics_set[i])
            
            # Almacenar el modelo
            results['model'].append(model)
            
            mae = val_metrics[0]
            if mae < best_score:
                best_fold = fold
                best_score = mae                
                best_model = model
                residual_model_=residual_model
                      
        self.model=best_model
        self.residual_model=residual_model_

        best_model_results_ = (test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3])
        best_model_results = (best_model, best_fold) + best_model_results_
        
        return results, best_model_results, best_model, residual_model_

    def predicter_corrected_2(self, X_test=None, y_test=None, residual_model=None):
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        if residual_model is None:
            residual_model = self.residual_model
        
        
        y_pred_test=self.predicter(X_test)
        correction = residual_model.predict(y_test.to_numpy().reshape(-1, 1))
        y_pred_adjusted = y_pred_test - correction
        
        return y_pred_adjusted




    def regression_metrics(self, y_true, y_pred):
        """
        Calcula las métricas de regresión: MAE, MSE, RMSE y R2.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    def best_hyper(self, opt_model, num_best=10, num_max=400):
        """
        Obtiene los mejores hiperparámetros para las mejores puntuaciones de validación cruzada dentro de los primeros num_max resultados.
       
        """
        results = opt_model.cv_results_
        errors = results['mean_test_score'][:num_max]  # Considerar solo los primeros num_max resultados
        best_idx = np.argsort(errors)[-num_best:]  # Obtener los índices de las mejores puntuaciones
        best_hypers = []

        for idx in best_idx:
            hyper = {}
            for param, value in results['params'][idx].items():
                hyper[param] = value
            best_hypers.append(hyper)

        # Invertir el orden para que el mejor esté en el índice 0
        best_hypers = best_hypers[::-1]

        return best_hypers

    
    def feature_importance_shap(self, X_test, model, random_seed=42):        
        try:
            self.explainer = shap.Explainer(model)
            shap_values = self.explainer.shap_values(X_test)
        except Exception as e:
            print("Fallo al usar shap.Explainer, intentando con shap.KernelExplainer:", e)
            try:
                np.random.seed(random_seed)
                self.explainer = shap.KernelExplainer(model.predict, shap.sample(self.X_train, 10), num_jobs=-1)
                shap_values = self.explainer.shap_values(X_test)
            except Exception as kernel_e:
                print("Fallo al usar shap.KernelExplainer:", kernel_e)
                return None, None 

        shap_sum = np.abs(shap_values).sum(axis=0)
        # Crear un diccionario para almacenar la suma de SHAP por característica
        shap_summary = {feature: shap_sum[i] for i, feature in enumerate(X_test.columns)}

        # Ordenar las características por su suma de SHAP
        shap_summary_sorted = sorted(shap_summary.items(), key=lambda x: x[1], reverse=True)

        # Imprimir el listado de importancia de características
        print("Importancia de características basada en suma de valores SHAP:")
        for feature, shap_sum in shap_summary_sorted:
            print(f"{feature}: {shap_sum}")
        
        return shap_values, shap_summary_sorted
       


    def shap_region(self, shap_summary_sorted, num_max=20):
        # Crear un diccionario para almacenar la suma de SHAP por región cerebral
        shap_por_region = {}

        # Recorrer la lista shap_summary_sorted
        for feature, shap_value in shap_summary_sorted[:num_max]:
            # Extraer la región cerebral (últimos dos textos separados por '_')
            region = feature.split('_')[-2] + '_' + feature.split('_')[-1]
            
            # Agregar la región cerebral al diccionario si no existe
            if region not in shap_por_region:
                shap_por_region[region] = 0.0
            
            # Sumar el valor SHAP al total de esa región cerebral
            shap_por_region[region] += shap_value
        
        max_value = max(shap_por_region.values())

        # Crear un diccionario para almacenar los valores normalizados
        resultado_normalizado = {}

        # Normalizar cada valor en el diccionario y almacenarlos en resultado_normalizado
        for region, suma_shap in shap_por_region.items():
            resultado_normalizado[region] = suma_shap / max_value

        # Ordenar shap_por_region y resultado_normalizado por valor descendente
        shap_por_region_sorted = {k: v for k, v in sorted(shap_por_region.items(), key=lambda item: item[1], reverse=True)}
        resultado_normalizado_sorted = {k: v for k, v in sorted(resultado_normalizado.items(), key=lambda item: item[1], reverse=True)}

        # Imprimir los valores normalizados ordenados
        for region, valor_normalizado in resultado_normalizado_sorted.items():
            print(f'{region}: {valor_normalizado:.6f}')

        return shap_por_region_sorted, resultado_normalizado_sorted

    
    def evaluacion_incremento_metricas(self, shap_values, n_iter=10,save_result=False):
        """
        Evalúa el impacto de añadir características basadas en su importancia, encontrando el mejor modelo para conjuntos de características incrementales y registrando las métricas de rendimiento.
        """
        shap_sum = np.abs(shap_values).sum(axis=0)
        # Crear un diccionario para almacenar la suma de SHAP por característica
        shap_summary = {feature: shap_sum[i] for i, feature in enumerate(self.X_test.columns)}

        # Ordenar las características por su suma de SHAP
        shap_summary_sorted = sorted(shap_summary.items(), key=lambda x: x[1], reverse=True)

        feature_names_list = []
        results_metricas={}

        print("Importancia de características basada en suma de valores SHAP:")
        for feature, shap_sum in shap_summary_sorted:            
            feature_names_list.append(feature)
            print("Lista de características hasta el momento:", feature_names_list)

            size_of_list = len(feature_names_list)
            print(f'Evaluación para {size_of_list} métrica(s)')

            X_train = self.X_train[feature_names_list]
            X_test = self.X_test[feature_names_list]        
            
            opt_model, parametros = self.search_best_model(X=X_train, n_iter_=n_iter)
            results, best_model_results_, best_model= self.trainer(X_train=X_train, X_test=X_test, params_=parametros)
            results_metricas[size_of_list-1] = best_model_results_
        
        return results_metricas
    
    
    