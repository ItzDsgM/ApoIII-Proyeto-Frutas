"""Modelos de Machine Learning tradicional y línea base (Fase 4 CRISP-DM).

Incluye:
- Línea base (``DummyClassifier`` mayoritaria y estratificada) exigida en Fase 5.
- Tres modelos clásicos distintos ajustados con ``GridSearchCV`` y validación cruzada
  agrupada (``StratifiedGroupKFold``, 5 pliegues) con ``return_train_score=True`` para
  diagnosticar sobreajuste (Rubrica Criterio 6):
  1. Máquina de Soporte Vectorial (SVM lineal y RBF).
  2. Random Forest.
  3. XGBoost (Gradient Boosted Trees).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import QUALITY_CLASSES, QUALITY_TO_IDX, RANDOM_SEED


def _macro_f1_scorer():
    return make_scorer(f1_score, labels=list(range(len(QUALITY_CLASSES))), average="macro", zero_division=0)


def train_classical_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Entrena línea base + SVM + Random Forest + XGBoost con ajuste por CV agrupada.

    ``y_train`` y ``y_val`` deben ser índices enteros en ``{0, 1, 2}`` según ``QUALITY_TO_IDX``.
    """
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scorer = _macro_f1_scorer()
    results: dict[str, dict[str, Any]] = {}

    # 1. Líneas base (DummyClassifier)
    for strat_key, label in (("most_frequent", "Baseline (Mayoritaria)"), ("stratified", "Baseline (Estratificada)")):
        t0 = time.perf_counter()
        dummy = DummyClassifier(strategy=strat_key, random_state=seed)
        dummy.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        tr_pred = dummy.predict(X_train)
        va_pred = dummy.predict(X_val)
        results[label] = {
            "estimator": dummy,
            "family": "baseline",
            "best_params": {"strategy": strat_key},
            "cv_train_macro_f1": float(f1_score(y_train, tr_pred, average="macro", zero_division=0)),
            "cv_val_macro_f1": float(f1_score(y_val, va_pred, average="macro", zero_division=0)),
            "train_macro_f1": float(f1_score(y_train, tr_pred, average="macro", zero_division=0)),
            "val_macro_f1": float(f1_score(y_val, va_pred, average="macro", zero_division=0)),
            "train_time_sec": round(elapsed, 3),
        }

    # 2. SVM (con StandardScaler dentro del pipeline para evitar fuga de datos)
    t0 = time.perf_counter()
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced", probability=True, random_state=seed)),
        ]
    )
    svm_grid = [
        {"svc__kernel": ["rbf"], "svc__C": [0.5, 1.0, 3.0, 10.0], "svc__gamma": ["scale", 0.005]},
        {"svc__kernel": ["linear"], "svc__C": [0.05, 0.2, 1.0]},
    ]
    svm_search = GridSearchCV(
        svm_pipe,
        svm_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    svm_search.fit(X_train, y_train, groups=groups_train)
    svm_time = time.perf_counter() - t0
    best_idx = int(svm_search.best_index_)
    svm_best = svm_search.best_estimator_
    results["SVM (Clásico)"] = {
        "estimator": svm_best,
        "family": "ml_clasico",
        "best_params": svm_search.best_params_,
        "cv_train_macro_f1": float(svm_search.cv_results_["mean_train_score"][best_idx]),
        "cv_val_macro_f1": float(svm_search.best_score_),
        "train_macro_f1": float(f1_score(y_train, svm_best.predict(X_train), average="macro", zero_division=0)),
        "val_macro_f1": float(f1_score(y_val, svm_best.predict(X_val), average="macro", zero_division=0)),
        "train_time_sec": round(svm_time, 3),
    }

    # 3. Random Forest
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 16, None],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"],
    }
    rf_search = GridSearchCV(
        rf,
        rf_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    rf_search.fit(X_train, y_train, groups=groups_train)
    rf_time = time.perf_counter() - t0
    best_idx = int(rf_search.best_index_)
    rf_best = rf_search.best_estimator_
    results["Random Forest (Clásico)"] = {
        "estimator": rf_best,
        "family": "ml_clasico",
        "best_params": rf_search.best_params_,
        "cv_train_macro_f1": float(rf_search.cv_results_["mean_train_score"][best_idx]),
        "cv_val_macro_f1": float(rf_search.best_score_),
        "train_macro_f1": float(f1_score(y_train, rf_best.predict(X_train), average="macro", zero_division=0)),
        "val_macro_f1": float(f1_score(y_val, rf_best.predict(X_val), average="macro", zero_division=0)),
        "train_time_sec": round(rf_time, 3),
    }

    # 4. XGBoost con pesos balanceados por clase
    t0 = time.perf_counter()
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(QUALITY_CLASSES),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )
    xgb_grid = {
        "n_estimators": [120, 220],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.12],
        "subsample": [0.85],
        "colsample_bytree": [0.8],
    }
    xgb_search = GridSearchCV(
        xgb,
        xgb_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=2,
        refit=True,
        return_train_score=True,
    )
    xgb_search.fit(X_train, y_train, groups=groups_train, sample_weight=sample_weights)
    xgb_time = time.perf_counter() - t0
    best_idx = int(xgb_search.best_index_)
    xgb_best = xgb_search.best_estimator_
    results["XGBoost (Clásico)"] = {
        "estimator": xgb_best,
        "family": "ml_clasico",
        "best_params": xgb_search.best_params_,
        "cv_train_macro_f1": float(xgb_search.cv_results_["mean_train_score"][best_idx]),
        "cv_val_macro_f1": float(xgb_search.best_score_),
        "train_macro_f1": float(f1_score(y_train, xgb_best.predict(X_train), average="macro", zero_division=0)),
        "val_macro_f1": float(f1_score(y_val, xgb_best.predict(X_val), average="macro", zero_division=0)),
        "train_time_sec": round(xgb_time, 3),
    }

    return results


def encode_labels(series) -> np.ndarray:
    """Convierte una secuencia de etiquetas textuales en índices enteros ``0..K-1``."""
    return np.array([QUALITY_TO_IDX[str(lbl)] for lbl in series], dtype=np.int64)
