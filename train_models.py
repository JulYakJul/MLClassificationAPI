import pandas as pd
import numpy as np
import warnings
import joblib
import os
import json
from pathlib import Path
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from catboost import CatBoostClassifier, Pool

from preprocessing import (
    remove_columns, 
    fill_missing_values, 
    get_feature_groups, 
    make_preprocess_pipeline
)

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def plot_learning_curve(estimator, X, y, title, filename, cv=3):
    """
    Построение кривой обучения для анализа переобучения/недообучения
    
    Args:
        estimator: обученная модель
        X: признаки
        y: целевая переменная
        title: название графика
        filename: имя файла для сохранения
        cv: количество фолдов для кросс-валидации
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, 
        cv=cv, 
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Обучающая выборка')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Валидационная выборка')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('Точность (Accuracy)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    final_train_score = train_mean[-1]
    final_test_score = test_mean[-1]
    gap = final_train_score - final_test_score
    
    print(f"\nАнализ кривой обучения для {title}:")
    print(f"   Точность на обучении: {final_train_score:.4f}")
    print(f"   Точность на валидации: {final_test_score:.4f}")
    print(f"   Разрыв (gap): {gap:.4f}")
    
    if gap > 0.1:
        print("ПЕРЕОБУЧЕНИЕ: модель слишком хорошо подогналась под обучающие данные")
    elif final_test_score < 0.6:
        print("НЕДООБУЧЕНИЕ: модель недостаточно сложна для данных")
    else:
        print("Хороший баланс между обучением и обобщением")


def save_model_and_metadata(model, model_name, metadata):
    """
    Сохранение модели и её метаданных
    
    Args:
        model: обученная модель
        model_name: имя модели
        metadata: метаданные (параметры, метрики)
    """
    
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f" Модель сохранена: {model_path}")
    
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f" Метаданные сохранены: {metadata_path}")


def evaluate_and_save(name, grid, X_train, y_train, X_test, y_test):
    """
    Оценка модели, построение кривой обучения и сохранение результатов
    
    Args:
        name: имя модели
        grid: обученный GridSearchCV объект
        X_train, y_train: обучающие данные
        X_test, y_test: тестовые данные
    Returns:
        best_model: лучшая модель
    """
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    train_acc = best.score(X_train, y_train)
    
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"{'='*60}")
    print("Лучшие параметры:", grid.best_params_)
    print(f"Точность (train): {train_acc:.4f}")
    print(f"Точность (test):  {acc:.4f}")
    print(f"F1 macro (test):  {f1m:.4f}")
    print(f"Разность (переобучение): {train_acc - acc:.4f}")
    
    plot_learning_curve(
        best, X_train, y_train,
        title=f"Кривая обучения: {name}",
        filename=f"{name.replace(' ', '_')}_learning_curve.png",
        cv=3
    )
    
    metadata = {
        "model_name": name,
        "best_params": grid.best_params_,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(acc),
        "f1_macro": float(f1m),
        "overfitting_gap": float(train_acc - acc)
    }
    
    save_model_and_metadata(best, name.replace(' ', '_'), metadata)
    
    return best


def train_decision_tree(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols):
    pre = make_preprocess_pipeline(numeric_cols, categorical_cols)
    
    pipe = Pipeline([
        ('pre', pre),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    
    param_grid = {
        'clf__max_depth': [3, 5, 7, 10], # максимальная глубина дерева - контролирует сложность модели
        'clf__criterion': ['gini', 'entropy'], # критерий разбиения - влияет на качество выбора лучшего признака. 
        # gini - чем меньше разнообразия классов, тем лучше (чище). 
        # entropy - мера неопределенности, стремится к уменьшению хаоса.
        'clf__min_samples_split': [2, 5, 10] # минимальное число образцов для разбиения - увелич./уменьшает переобучение
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1,
                        scoring={'acc': 'accuracy', 'f1': 'f1_macro'}, refit='f1')
    
    grid.fit(X_train, y_train)
    
    return evaluate_and_save('Decision_Tree', grid, X_train, y_train, X_test, y_test)

def train_knn(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols):
    """
    Обучение KNN
    """
    pre = make_preprocess_pipeline(numeric_cols, categorical_cols)
    
    pipe = Pipeline([
        ('pre', pre),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', KNeighborsClassifier())
    ])
    
    param_grid = {
        'clf__n_neighbors': [5, 7, 9, 11, 13, 15, 17, 20, 25], # большее число соседей для сглаживания решений
        'clf__weights': ['uniform', 'distance'], # 'distance' - ближние соседи влияют больше/ 'uniform' - все одинаково
        'clf__metric': ['euclidean', 'manhattan', 'minkowski'], # euclidean - стандартное расстояние от точки до точки.
         # manhattan - сумма абсолютных разниц по каждой оси.
         # minkowski - обобщение евклидова и манхэттенского расстояний.
        'clf__p': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        n_jobs=-1,
        scoring={'acc': 'accuracy', 'f1': 'f1_macro'},
        refit='f1',
        verbose=1
    )
    grid.fit(X_train, y_train)
    
    return evaluate_and_save('KNN', grid, X_train, y_train, X_test, y_test)

def train_logistic_regression(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols):
    """
    Обучение логистической регрессии с полиномиальными признаками
    """
    pre = make_preprocess_pipeline(numeric_cols, categorical_cols)
    
    pipe = Pipeline([
        ('pre', pre),
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression( 
            solver='saga', # оптимизатор для больших наборов данных и L1/L2 регуляризации
            max_iter=1000, # максимальное число итераций для сходимости
            class_weight='balanced', # балансировка классов для борьбы с дисбалансом
            random_state=42
        ))
    ])
    
    param_grid = {
        'poly__degree': [1, 2], # степень полинома - 1 (линейная), 2 (квадратичная)
        'clf__C': [0.1, 1.0, 10.0], # обратная величина регуляризации (штраф за сложность) - меньшее значение сильнее регуляризирует
    }
    
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1,
                        scoring={'acc': 'accuracy', 'f1': 'f1_macro'}, refit='f1')
    
    grid.fit(X_train, y_train)
    
    return evaluate_and_save('Logistic_Regression', grid, X_train, y_train, X_test, y_test)


def train_mlp(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols):
    """
    Обучение MLP
    """
    pre = make_preprocess_pipeline(numeric_cols, categorical_cols)
    
    pipe = Pipeline([
        ('pre', pre),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', MLPClassifier(
            early_stopping=True, # ранняя остановка для предотвращения переобучения
            validation_fraction=0.2, # доля валидационной выборки для ранней остановки
            n_iter_no_change=20, # количество итераций без улучшения для остановки
            random_state=42
        ))
    ])
    
    param_grid = {
        'clf__hidden_layer_sizes': [(50,), (100,), (50, 25)],  # слои и нейроны
        'clf__alpha': [0.01, 0.05, 0.1], # коэффициент регуляризации L2 - помогает бороться с переобучением
        'clf__learning_rate_init': [0.001, 0.01],  # скорость обучения
        'clf__batch_size': [128, 256],  # размер батча
        'clf__activation': ['relu', 'tanh'],  # функции активации. relu - стандарт для глубоких сетей, tanh - для более гладких переходов
        'clf__max_iter': [500, 1000]  # итерации
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        n_jobs=-1,
        scoring={'acc': 'accuracy', 'f1': 'f1_macro'},
        refit='f1',
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    return evaluate_and_save('MLP', grid, X_train, y_train, X_test, y_test)


def plot_catboost_learning_curve(model_params, X_train, y_train, X_test, y_test, categorical_indices, title, filename):
    """
    CatBoost
    """
    plt.figure(figsize=(10, 6))

    train_sizes = np.linspace(0.1, 1.0, 8)
    train_scores = []
    valid_scores = []

    test_pool = Pool(X_test, y_test, cat_features=categorical_indices)

    for frac in train_sizes:
        size = int(len(X_train) * frac)

        X_part = X_train[:size]
        y_part = y_train[:size]

        train_pool = Pool(X_part, y_part, cat_features=categorical_indices)

        model = CatBoostClassifier(**model_params)
        model.fit(train_pool, verbose=False)

        pred_train = model.predict(train_pool)
        train_acc = accuracy_score(y_part, pred_train)
        train_scores.append(train_acc)

        pred_valid = model.predict(test_pool)
        valid_acc = accuracy_score(y_test, pred_valid)
        valid_scores.append(valid_acc)

    plt.plot(train_sizes, train_scores, label="Train accuracy", marker='o')
    plt.plot(train_sizes, valid_scores, label="Validation accuracy", marker='o')

    plt.title(title)
    plt.xlabel("Train size (%)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print("\nLearning curve saved:", PLOTS_DIR / filename)

def train_catboost(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols):
    """
    Обучение CatBoost
    """
    print("\n=== Обучение CatBoost ===")

    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)

    model = CatBoostClassifier(
        loss_function='MultiClass', # многоклассовая классификация
        eval_metric='Accuracy', # метрика для оценки качества
        iterations=1000, # количество деревьев
        depth=8, # глубина деревьев
        learning_rate=0.05, # скорость обучения
        random_seed=42, 
        verbose=200 # вывод прогресса каждые 200 итераций
    )

    model.fit(train_pool, eval_set=test_pool)

    y_pred = model.predict(test_pool).flatten().astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')

    print(f"\nCatBoost Accuracy: {acc:.4f}")
    print(f"CatBoost F1 Macro:  {f1m:.4f}")

    save_model_and_metadata(model, "CatBoost", {
        "model_name": "CatBoost",
        "accuracy": float(acc),
        "f1_macro": float(f1m),
    })

    plot_catboost_learning_curve(
        {
            "loss_function": "MultiClass",
            "eval_metric": "Accuracy",
            "iterations": 300,
            "depth": 8,
            "learning_rate": 0.05,
            "random_seed": 42
        },
        X_train, y_train,
        X_test, y_test,
        cat_indices,
        title="Learning Curve — CatBoost",
        filename="CatBoost_learning_curve.png"
    )

    return model


def main():
    print("="*60)
    print("ОБУЧЕНИЕ ML МОДЕЛЕЙ")
    print("="*60)
    
    df = pd.read_csv("new_retail_data.csv")
    print(f"\nЗагружено данных: {df.shape[0]} строк, {df.shape[1]} колонок")
    
    target_col = "Product_Category"
    df_proc = remove_columns(df)
    df_proc = fill_missing_values(df_proc)
    
    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nРаспределение классов:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([cls])[0]
        print(f"   Класс {cls} ({class_name}): {count} samples ({count/len(y_encoded)*100:.1f}%)")
    
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    print(f"\nLabelEncoder сохранен: {MODELS_DIR / 'label_encoder.joblib'}")
    print(f"Классы: {dict(enumerate(label_encoder.classes_))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nРазделение данных:")
    print(f"   Train: {X_train.shape[0]} образцов")
    print(f"   Test:  {X_test.shape[0]} образцов")
    
    numeric_cols, categorical_cols = get_feature_groups(df_proc, target_col)
    print(f"\nПризнаки:")
    print(f"   Числовых: {len(numeric_cols)}")
    print(f"   Категориальных: {len(categorical_cols)}")
    print(f"   Числовые признаки: {numeric_cols}")
    print(f"   Категориальные признаки: {categorical_cols}")
    
    results = {}
    
    print("\n" + "="*60)
    print("Начало обучения моделей...")
    print("="*60)
    
    # Decision Tree
    print("\n--- Обучение Decision Tree ---")
    model_dt = train_decision_tree(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
    results['Decision_Tree'] = accuracy_score(y_test, model_dt.predict(X_test))
    
    # KNN
    print("\n--- Обучение KNN ---")
    model_knn = train_knn(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
    results['KNN'] = accuracy_score(y_test, model_knn.predict(X_test))
    
    # Logistic Regression
    print("\n--- Обучение Logistic Regression ---")
    model_lr = train_logistic_regression(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
    results['Logistic_Regression'] = accuracy_score(y_test, model_lr.predict(X_test))
    
    # MLP
    print("\n--- Обучение MLP ---")
    model_mlp = train_mlp(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
    results['MLP'] = accuracy_score(y_test, model_mlp.predict(X_test))
    
    # CatBoost
    print("\n--- Обучение CatBoost ---")
    model_cat = train_catboost(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
    results["CatBoost"] = accuracy_score(y_test, model_cat.predict(X_test))

    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ")
    print("="*60)
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {acc:.4f}")
    
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    print(f"\n Лучшая модель: {best_model_name} с точностью {best_accuracy:.4f}")
    
    summary = {
        "models": results,
        "best_model": best_model_name,
        "best_accuracy": float(best_accuracy),
        "target_classes": label_encoder.classes_.tolist(),
        "class_distribution": dict(zip([str(cls) for cls in unique], counts.tolist())),
        "dataset_info": {
            "total_samples": len(df_proc),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols)
        }
    }
    
    with open(MODELS_DIR / "training_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n Все модели и результаты сохранены в директории: {MODELS_DIR}")
    print(f" Графики кривых обучения сохранены в директории: {PLOTS_DIR}")
    
    print(f"\nДополнительный анализ лучшей модели ({best_model_name}):")
    if best_model_name == 'KNN':
        best_model = model_knn
    elif best_model_name == 'MLP':
        best_model = model_mlp
    
    y_pred = best_model.predict(X_test)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print("F1-score по классам:")
    for i, (cls, f1) in enumerate(zip(label_encoder.classes_, f1_per_class)):
        print(f"   {cls}: {f1:.4f}")


if __name__ == "__main__":
    main()