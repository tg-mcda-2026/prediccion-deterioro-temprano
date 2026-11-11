"""
Modelos del protocolo: construcción de los 7 algoritmos (logística, tarjeta WoE, random forest, LightGBM, XGBoost,
CatBoost y EBM), su espacio de búsqueda en Optuna y la recalibración del intercepto.

Qué recibe cada algoritmo:
  - Logística, random forest, LightGBM y XGBoost: numéricas imputadas con la mediana (escaladas en la logística);
    categóricas de pocos niveles con una columna por categoría (las de menos de 30 casos se agrupan);
    categóricas de muchos niveles (ciudad, departamento) codificadas con la tasa del evento, ajustada dentro del pliegue.
  - Tarjeta WoE: discretización supervisada y peso de la evidencia (optbinning) + regresión logística.
  - CatBoost: categóricas en forma nativa (codificación ordenada).
  - EBM: variables originales; el modelo maneja las categóricas.
Las listas NUMERICAS, CAT_BAJA y CAT_ALTA las fija quien importa el módulo (protocolo.py).
"""
import warnings

import numpy as np
import optuna
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from optbinning import BinningProcess
from scipy.optimize import minimize_scalar
from sklearn import set_config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
set_config(transform_output="pandas")

NUMERICAS, CAT_BAJA, CAT_ALTA = [], [], []  # las fija protocolo.py

SEMILLA = 42


N_TRIALS = {"logistica": 25, "scorecard": 20, "random_forest": 25, "lightgbm": 40, "xgboost": 40,
            "catboost": 15, "ebm": 10}


def preprocesador(variables, escalar):
    num = [v for v in variables if v in NUMERICAS]
    baja = [v for v in variables if v in CAT_BAJA]
    alta = [v for v in variables if v in CAT_ALTA]
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    transformadores = [("num", Pipeline(pasos_num), num)]
    if baja:
        transformadores.append(("baja", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist",
                                                      sparse_output=False), baja))
    if alta:
        alta_pasos = [("te", TargetEncoder(target_type="binary", random_state=SEMILLA))]
        if escalar:
            alta_pasos.append(("escalar", StandardScaler()))
        transformadores.append(("alta", Pipeline(alta_pasos), alta))
    return ColumnTransformer(transformadores, verbose_feature_names_out=False)


class CatBoostSklearn(ClassifierMixin, BaseEstimator):
    """CatBoost modifica `cat_features` al construirse y sklearn no puede clonarlo; este envoltorio sí."""

    def __init__(self, cat_features=(), iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg

    def fit(self, X, y):
        self.modelo_ = CatBoostClassifier(
            cat_features=list(self.cat_features), iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, l2_leaf_reg=self.l2_leaf_reg, verbose=0, random_seed=SEMILLA,
            thread_count=-1, allow_writing_files=False).fit(X, y)
        self.classes_ = self.modelo_.classes_
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)

    def predict(self, X):
        return self.modelo_.predict(X)


class Promedio(ClassifierMixin, BaseEstimator):
    """Ensamble: promedio de probabilidades de modelos ya ajustados."""

    def __init__(self, modelos=()):
        self.modelos = modelos
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.modelos], axis=0)


def seleccion(variables):
    return ColumnTransformer([("todo", "passthrough", variables)], verbose_feature_names_out=False)


def construir(nombre, params, variables):
    cats = [v for v in variables if v in CAT_BAJA + CAT_ALTA]
    if nombre == "scorecard":
        params = dict(params)
        binning = BinningProcess(variable_names=list(variables), categorical_variables=cats,
                                 max_n_bins=params.pop("max_n_bins"), min_bin_size=params.pop("min_bin_size"))
        return Pipeline([("sel", seleccion(variables)), ("woe", binning),
                         ("modelo", LogisticRegression(**params, max_iter=5000))])
    if nombre == "ebm":
        modelo = ExplainableBoostingClassifier(**params, outer_bags=4, n_jobs=-1, random_state=SEMILLA)
        return Pipeline([("sel", seleccion(variables)), ("modelo", modelo)])
    if nombre == "catboost":
        return Pipeline([("sel", seleccion(variables)), ("modelo", CatBoostSklearn(cat_features=tuple(cats), **params))])
    if nombre == "logistica":
        modelo = LogisticRegression(**params, max_iter=5000)
    elif nombre == "random_forest":
        modelo = RandomForestClassifier(**params, n_estimators=500, n_jobs=-1, random_state=SEMILLA)
    elif nombre == "lightgbm":
        modelo = LGBMClassifier(**params, subsample_freq=1, verbose=-1, random_state=SEMILLA, n_jobs=-1)
    elif nombre == "xgboost":
        modelo = XGBClassifier(**params, tree_method="hist", eval_metric="logloss",
                               random_state=SEMILLA, n_jobs=-1)
    return Pipeline([("prep", preprocesador(variables, escalar=nombre == "logistica")), ("modelo", modelo)])


def espacio(nombre, t):
    if nombre == "logistica":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if nombre == "random_forest":
        return {"max_depth": t.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 100),
                "max_features": t.suggest_float("max_features", 0.2, 0.8)}
    if nombre == "lightgbm":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 800),
                "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves": t.suggest_int("num_leaves", 4, 32),
                "min_child_samples": t.suggest_int("min_child_samples", 10, 120),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if nombre == "xgboost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 800),
                "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 6),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 30, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if nombre == "scorecard":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True),
                "max_n_bins": t.suggest_int("max_n_bins", 3, 8),
                "min_bin_size": t.suggest_float("min_bin_size", 0.03, 0.15)}
    if nombre == "ebm":
        return {"learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "max_leaves": t.suggest_int("max_leaves", 2, 4),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 60),
                "interactions": t.suggest_int("interactions", 0, 10),
                "max_bins": t.suggest_int("max_bins", 32, 256)}
    if nombre == "catboost":
        return {"iterations": t.suggest_int("iterations", 200, 800),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": t.suggest_int("depth", 3, 7),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 20, log=True)}


def recalibrar_intercepto(y, p):
    """Desplazamiento del logit que minimiza la log-loss (ajusta el nivel sin cambiar el orden del riesgo)."""
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    ll = lambda a: log_loss(y, 1 / (1 + np.exp(-(logit + a))), labels=[0, 1])
    return float(minimize_scalar(ll, bounds=(-5, 5), method="bounded").x)


def aplicar_intercepto(p, a):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(logit + a)))
