"""
Функции для экспериментов по классификации с использованием признаков MyModel.

Функции:
- get_classifier: Получение классификатора по имени
- evaluate_classifier: Обучение и оценка классификатора
- extract_mymodel_features: Извлечение признаков MyModel
- compare_all_classifiers_with_mymodel_features: Сравнение классификаторов
- plot_all_sphere_types_comparison: Визуализация сравнения по типам сфер
- generate_sphere_dataset: Генерация датасетов на основе сфер
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from models import MyModel


# =============================================================================
# Генерация данных
# =============================================================================

def _sample_unit_directions(n: int, d: int, rng) -> np.ndarray:
    """Генерирует случайные направления на единичной сфере."""
    z = rng.normal(size=(n, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return z / norms


def _sample_radii(n: int, d: int, a: float, b: float, rng) -> np.ndarray:
    """Генерирует равномерные радиусы в d-мерной оболочке [a, b]."""
    u = rng.random(n)
    a_d = a**d
    b_d = b**d
    r = (u * (b_d - a_d) + a_d) ** (1.0 / d)
    return r


def generate_sphere_dataset(
    n_cases: int,
    n_controls: int,
    n_dims: int = 10,
    model: str = "ideal",
    noise_dims: int = 50,
    broken_fraction: float = 0.5,
    r_ctrl: Tuple[float, float] = (0.01, 0.5),
    r_case: Tuple[float, float] = (0.5, 1.0),
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генератор датасетов на основе сфер/полусфер/шумных/сломанных.

    Parameters
    ----------
    n_cases : int
        Количество объектов класса 1 (cases)
    n_controls : int
        Количество объектов класса 0 (controls)
    n_dims : int
        Размерность данных --- РЕЗУЛЬТИРУЮЩАЯ!!!
    model : str
        Тип модели: "ideal", "noisy", "broken", "hemisphere", "noisy_hemisphere"
    noise_dims : int
        Количество шумовых измерений (для noisy моделей)
    broken_fraction : float
        Доля "сломанных" координат (для broken модели)
    r_ctrl : tuple
        Диапазон радиусов для controls
    r_case : tuple
        Диапазон радиусов для cases
    random_state : int
        Для воспроизводимости

    Returns
    -------
    X : np.ndarray
        Матрица признаков
    y : np.ndarray
        Метки классов
    radii : np.ndarray
        Радиусы точек
    """

    assert n_dims > noise_dims, "Должно быть что-то кроме шума!"

    rng = np.random.default_rng(random_state)
    N = n_cases + n_controls

    dirs = _sample_unit_directions(N, n_dims, rng)

    r_controls = _sample_radii(n_controls, n_dims, r_ctrl[0], r_ctrl[1], rng)
    r_cases = _sample_radii(n_cases, n_dims, r_case[0], r_case[1], rng)

    radii = np.concatenate([r_controls, r_cases], axis=0)
    X_base = dirs * radii.reshape(-1, 1)

    y = np.concatenate(
        [np.zeros(n_controls, dtype=int), np.ones(n_cases, dtype=int)], axis=0
    )

    if model == "ideal":
        X = X_base.copy()

    elif model == "hemisphere":
        X = X_base.copy()
        mask = X[:, 0] < 0
        X[mask] *= -1
    elif model == "noisy":
        n_sphere_dims = n_dims - noise_dims
        X_sphere = X_base[:, :n_sphere_dims]
        noise = rng.uniform(-1, 1, size=(N, noise_dims))
        X = np.hstack([X_sphere, noise])

    elif model == "noisy_hemisphere":
        X = X_base.copy()
        mask = X[:, 0] < 0
        X[mask] *= -1
        n_sphere_dims = n_dims - noise_dims
        X_sphere = X[:, :n_sphere_dims]
        noise = rng.uniform(-1, 1, size=(N, noise_dims))
        X = np.hstack([X_sphere, noise])

    elif model == "broken":
        X = X_base.copy()
        k = int(np.floor(n_dims * broken_fraction))
        if k != 0:
            X[:, -k:] = rng.uniform(-1, 1, size=(N, k))

    else:
        raise ValueError(f"Unknown model: {model}")

    return X, y, radii


# =============================================================================
# Классификаторы
# =============================================================================

def get_classifier(name: str, random_state: int = 42):
    """
    Возвращает классификатор по имени.

    Parameters
    ----------
    name : str
        Название классификатора
    random_state : int
        Для воспроизводимости

    Returns
    -------
    classifier
        Экземпляр классификатора
    """
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "SVC": SVC(kernel='rbf', random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    return classifiers.get(name, LogisticRegression(max_iter=1000, random_state=random_state))


def evaluate_classifier(
    clf,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    clf_name: str,
    feature_type: str
) -> Dict:
    """
    Обучает и оценивает классификатор.

    Parameters
    ----------
    clf : classifier
        Классификатор
    X_train, X_test : np.ndarray
        Признаки
    y_train, y_test : np.ndarray
        Метки
    clf_name : str
        Название классификатора
    feature_type : str
        Тип признаков

    Returns
    -------
    result : dict
        Словарь с метриками
    """
    # Стандартизация признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение
    clf.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred = clf.predict(X_test_scaled)
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {
        "classifier": clf_name,
        "feature_type": feature_type,
        "accuracy": acc,
        "f1_score": f1,
    }


def extract_mymodel_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    d: int,
    k: int,
    extraction_type: str = "old" # "old|full|full_aggregated"  
) -> Tuple[np.ndarray, np.ndarray, MyModel]:
    """
    Обучает MyModel и извлекает признаки для train и test.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Данные
    d : int
        Размерность
    k : int
        Параметр k для MyModel

    Returns
    -------
    X_train_features : np.ndarray
        Признаки для train (n_train, d * (k+1))
    X_test_features : np.ndarray
        Признаки для test (n_test, d * (k+1))
    mymodel : MyModel
        Обученная модель
    """
    # Обучаем MyModel на train
    mymodel = MyModel(d=d, k=k)
    mymodel.fit_parallel(X_train, n_jobs=7)
    
    # # Извлекаем признаки
    # # get_feature_matrix возвращает (n, d, k+1), нужно развернуть в (n, d*(k+1))
    # train_features_3d = mymodel.get_feature_matrix(X_train)
    # test_features_3d = mymodel.get_feature_matrix(X_test)
    
    # # Разворачиваем в 2D: (n, d, k+1) -> (n, d * (k+1))
    # n_train = train_features_3d.shape[0]
    # n_test = test_features_3d.shape[0]
    
    # X_train_features = train_features_3d.reshape(n_train, -1)
    # X_test_features = test_features_3d.reshape(n_test, -1)


    # Извлекаем признаки
    # get_feature_matrix возвращает (n, d, k+1), нужно развернуть в (n, d*(k+1))

    func_to_call = mymodel.get_feature_matrix

    if extraction_type == "full":
        func_to_call = mymodel.get_feature_matrix_full
    
    if extraction_type == "full_aggregated":
        func_to_call = mymodel.get_feature_matrix_full_aggregated

    X_train_features, _ = func_to_call(X_train)
    X_test_features, _ = func_to_call(X_test)
    
    return X_train_features, X_test_features, mymodel


# =============================================================================
# Сравнение классификаторов
# =============================================================================

def compare_all_classifiers_with_mymodel_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_mymodel: np.ndarray,
    X_test_mymodel: np.ndarray,
    classifiers_list: List[str],
    d: int = None,
    k: int = None,
    random_state: int = 42,
    figsize: Tuple[int, int] = (14, 7),
    save_path: str = "all_classifiers_comparison.png"
) -> pd.DataFrame:
    """
    Сравнивает все классификаторы на исходных признаках и на признаках MyModel.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Исходные признаки (Raw X)
    y_train, y_test : np.ndarray
        Метки классов
    X_train_mymodel, X_test_mymodel : np.ndarray
        Признаки из MyModel.get_feature_matrix()
    classifiers_list : list
        Список названий классификаторов
    d : int
        Размерность данных (для отображения в заголовке)
    k : int
        Параметр k для MyModel (для отображения в заголовке)
    random_state : int
        Для воспроизводимости
    figsize : tuple
        Размер фигуры
    save_path : str
        Путь для сохранения графика

    Returns
    -------
    results_df : pd.DataFrame
        Таблица с результатами
    """
    results = []
    
    for clf_name in classifiers_list:
        # 1. На исходных признаках (Raw X)
        clf_raw = get_classifier(clf_name, random_state=random_state)
        result_raw = evaluate_classifier(
            clf_raw, X_train, X_test, y_train, y_test,
            clf_name=clf_name,
            feature_type="Raw X"
        )
        results.append(result_raw)
        
        # 2. На признаках MyModel
        clf_mymodel = get_classifier(clf_name, random_state=random_state)
        result_mymodel = evaluate_classifier(
            clf_mymodel, X_train_mymodel, X_test_mymodel, y_train, y_test,
            clf_name=clf_name,
            feature_type="MyModel Features"
        )
        results.append(result_mymodel)
    
    results_df = pd.DataFrame(results)
    
    # =========================================================================
    # Построение графика
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(classifiers_list))
    width = 0.35
    
    # Данные для графиков
    acc_raw = results_df[results_df['feature_type'] == 'Raw X']['accuracy'].values
    acc_mymodel = results_df[results_df['feature_type'] == 'MyModel Features']['accuracy'].values
    
    f1_raw = results_df[results_df['feature_type'] == 'Raw X']['f1_score'].values
    f1_mymodel = results_df[results_df['feature_type'] == 'MyModel Features']['f1_score'].values
    
    # Цвета
    color_raw = '#3498db'      # синий
    color_mymodel = '#e74c3c'  # красный
    
    # ---- График 1: Accuracy ----
    ax1 = axes[0]
    bars1_raw = ax1.bar(x - width/2, acc_raw, width, label='Raw X', color=color_raw, alpha=0.85)
    bars1_mymodel = ax1.bar(x + width/2, acc_mymodel, width, label='MyModel Features', color=color_mymodel, alpha=0.85)
    
    ax1.set_xlabel('Классификатор', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Сравнение по Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classifiers_list, rotation=30, ha='right')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bars1_raw:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars1_mymodel:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # ---- График 2: F1-score ----
    ax2 = axes[1]
    bars2_raw = ax2.bar(x - width/2, f1_raw, width, label='Raw X', color=color_raw, alpha=0.85)
    bars2_mymodel = ax2.bar(x + width/2, f1_mymodel, width, label='MyModel Features', color=color_mymodel, alpha=0.85)
    
    ax2.set_xlabel('Классификатор', fontsize=12)
    ax2.set_ylabel('F1-score', fontsize=12)
    ax2.set_title('Сравнение по F1-score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classifiers_list, rotation=30, ha='right')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bars2_raw:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2_mymodel:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Формируем заголовок с информацией о параметрах
    n_raw_features = X_train.shape[1]
    n_mymodel_features = X_train_mymodel.shape[1]
    
    info_parts = []
    if d is not None:
        info_parts.append(f'd={d}')
    if k is not None:
        info_parts.append(f'k={k}')
    info_parts.append(f'Raw features: {n_raw_features}')
    info_parts.append(f'MyModel features: {n_mymodel_features}')
    
    title = 'Сравнение классификаторов: Raw X vs MyModel Features\n(' + ', '.join(info_parts) + ')'
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ График сохранён: {save_path}")
    
    return results_df


def plot_all_sphere_types_comparison(
    all_results_dict: Dict[str, pd.DataFrame],
    classifiers_list: List[str],
    d: int = None,
    k: int = None,
    n_raw_features: int = None,
    n_mymodel_features: int = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: str = "all_sphere_types_comparison.png"
):
    """
    Строит сводный график сравнения Raw X vs MyModel Features для всех типов сфер.

    Parameters
    ----------
    all_results_dict : dict
        Словарь {model_type: DataFrame с результатами}
    classifiers_list : list
        Список классификаторов
    d : int
        Размерность данных
    k : int
        Параметр k для MyModel
    n_raw_features : int
        Количество исходных признаков
    n_mymodel_features : int
        Количество признаков MyModel
    figsize : tuple
        Размер фигуры
    save_path : str
        Путь для сохранения графика
    """
    sphere_types = list(all_results_dict.keys())
    n_types = len(sphere_types)
    
    # Создаём фигуру с 2 рядами (Accuracy и F1) и n_types колонками
    fig, axes = plt.subplots(2, n_types, figsize=figsize, sharey='row')
    
    x = np.arange(len(classifiers_list))
    width = 0.35
    
    color_raw = '#3498db'      # синий
    color_mymodel = '#e74c3c'  # красный
    
    for col_idx, model_type in enumerate(sphere_types):
        df = all_results_dict[model_type]
        
        acc_raw = df[df['feature_type'] == 'Raw X']['accuracy'].values
        acc_mymodel = df[df['feature_type'] == 'MyModel Features']['accuracy'].values
        
        f1_raw = df[df['feature_type'] == 'Raw X']['f1_score'].values
        f1_mymodel = df[df['feature_type'] == 'MyModel Features']['f1_score'].values
        
        # ---- Accuracy (верхний ряд) ----
        ax_acc = axes[0, col_idx]
        ax_acc.bar(x - width/2, acc_raw, width, label='Raw X', color=color_raw, alpha=0.85)
        ax_acc.bar(x + width/2, acc_mymodel, width, label='MyModel', color=color_mymodel, alpha=0.85)
        ax_acc.set_title(f'{model_type}', fontsize=12, fontweight='bold')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(classifiers_list, rotation=45, ha='right', fontsize=8)
        ax_acc.set_ylim(0, 1.05)
        ax_acc.grid(axis='y', alpha=0.3)
        
        if col_idx == 0:
            ax_acc.set_ylabel('Accuracy', fontsize=11)
        if col_idx == n_types - 1:
            ax_acc.legend(fontsize=8, loc='lower right')
        
        # ---- F1-score (нижний ряд) ----
        ax_f1 = axes[1, col_idx]
        ax_f1.bar(x - width/2, f1_raw, width, label='Raw X', color=color_raw, alpha=0.85)
        ax_f1.bar(x + width/2, f1_mymodel, width, label='MyModel', color=color_mymodel, alpha=0.85)
        ax_f1.set_xticks(x)
        ax_f1.set_xticklabels(classifiers_list, rotation=45, ha='right', fontsize=8)
        ax_f1.set_ylim(0, 1.05)
        ax_f1.grid(axis='y', alpha=0.3)
        
        if col_idx == 0:
            ax_f1.set_ylabel('F1-score', fontsize=11)
    
    # Формируем заголовок с информацией о параметрах
    info_parts = []
    if d is not None:
        info_parts.append(f'd={d}')
    if k is not None:
        info_parts.append(f'k={k}')
    if n_raw_features is not None:
        info_parts.append(f'Raw features: {n_raw_features}')
    if n_mymodel_features is not None:
        info_parts.append(f'MyModel features: {n_mymodel_features}')
    
    if info_parts:
        title = f'Сравнение классификаторов: Raw X vs MyModel Features\n(по типам генерации данных, {", ".join(info_parts)})'
    else:
        title = 'Сравнение классификаторов: Raw X vs MyModel Features\n(по типам генерации данных)'
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ График сохранён: {save_path}")


# =============================================================================
# Список базовых классификаторов
# =============================================================================

BASELINE_CLASSIFIERS = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "SVC",
    "KNN",
]
