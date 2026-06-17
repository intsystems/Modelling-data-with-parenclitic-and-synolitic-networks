"""
Модели для экспериментов с k-order зависимостями Gaussian распределений.

Классы:
- KOrderDependencyGaussian: Истинная модель с k-order зависимостями
- MyModel: Параметрическая модель с разложением плотности в ряд
"""

import numpy as np
from scipy import linalg
from itertools import combinations
from math import comb


class KOrderDependencyGaussian:
    """
    Многомерное нормальное распределение с управляемой структурой зависимостей
    на основе Cholesky-разложения.
    
    Ковариационная матрица строится через Cholesky-фактор L:
    - Σ = L @ L.T
    - Элементы L[i, j] для j < i затухают по мере удаления от диагонали:
      L[i, j] = decay^(i - j - 1) если (i - j) ≤ k, иначе 0
    
    Это обеспечивает:
    - Положительную определённость ковариационной матрицы (гарантировано)
    - Управляемую структуру зависимостей: x_i зависит от x_{i-1}, ..., x_{i-k}
    - Затухание корреляций с расстоянием

    Parameters
    ----------
    d : int
        Размерность распределения (число переменных)
    k : int
        Порядок зависимости (0 = независимость, k = d-1 = полная зависимость)
    decay : float
        Фактор затухания для элементов Cholesky-матрицы (0 < decay < 1)
    sigma : float
        Диагональные элементы L (управляет общей дисперсией)
    mu_min : float
        Минимальное значение для случайных средних
    mu_max : float
        Максимальное значение для случайных средних
    random_state : int, optional
        Случайное состояние для воспроизводимости
    """

    def __init__(
        self,
        d,
        k,
        decay=0.3,
        sigma=1.0,
        mu_min=-1.0,
        mu_max=1.0,
        random_state=None,
    ):
        self.d = d
        self.k = min(k, d - 1)
        self.decay = decay
        self.sigma = sigma
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        
        # Генерируем случайные средние для каждой переменной
        self.mu = self.rng.uniform(mu_min, mu_max, size=d)

        # Строим ковариационную матрицу через Cholesky
        self._build_covariance_and_cholesky()

    def _build_covariance_and_cholesky(self):
        """
        Строим нижнетреугольную матрицу L (Cholesky-фактор) 
        и ковариационную матрицу Σ = L @ L.T
        """
        d, k = self.d, self.k

        L = np.zeros((d, d))

        for i in range(d):
            # Диагональ: L[i, i] = sigma
            L[i, i] = self.sigma

            # Поддиагональные элементы: L[i, j] для j < i
            for j in range(max(0, i - k), i):
                distance = i - j
                L[i, j] = self.decay ** (distance - 1) * self.sigma * 0.5

        self.L = L
        self.cov = L @ L.T
        
        # Предвычисляем L^{-1} для вычисления log_prob
        self.L_inv = linalg.solve_triangular(L, np.eye(d), lower=True)
        
        # log|det(Σ)| = 2 * sum(log(L[i,i]))
        self.log_det_cov = 2 * np.sum(np.log(np.diag(L)))

    def sample(self, n=1):
        """
        Генерация n сэмплов из распределения.
        
        X = μ + L @ Z, где Z ~ N(0, I)
        """
        Z = self.rng.randn(n, self.d)
        X = self.mu + Z @ self.L.T
        return X

    def log_prob(self, X):
        """
        Вычисление log p(x) для каждого сэмпла.
        
        log p(x) = -d/2 * log(2π) - 1/2 * log|Σ| - 1/2 * (x - μ)^T Σ^{-1} (x - μ)
        """
        X = np.atleast_2d(X)
        n, d = X.shape
        
        if d != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {d}")

        centered = X - self.mu  # (n, d)
        
        # Σ^{-1} = (L @ L.T)^{-1} = L^{-T} @ L^{-1}
        # (x - μ)^T Σ^{-1} (x - μ) = || L^{-1} (x - μ) ||^2
        transformed = centered @ self.L_inv.T  # (n, d)
        mahal_sq = np.sum(transformed ** 2, axis=1)
        
        log_p = (
            -0.5 * d * np.log(2 * np.pi)
            - 0.5 * self.log_det_cov
            - 0.5 * mahal_sq
        )
        
        return log_p if n > 1 else log_p[0]

    def get_covariance_matrix(self):
        """Возвращает ковариационную матрицу Σ"""
        return self.cov

    def get_cholesky_factor(self):
        """Возвращает Cholesky-фактор L"""
        return self.L

    def get_mean(self):
        """Возвращает вектор средних μ"""
        return self.mu

    def get_variance(self):
        """Возвращает вектор дисперсий (диагональ ковариационной матрицы)"""
        return np.diag(self.cov)

    def __repr__(self):
        return f"KOrderDependencyGaussian(d={self.d}, k={self.k}, decay={self.decay}, sigma={self.sigma})"


def _estimate_regressor_params(regressor, r):
    """
    Оценить "эффективное число параметров" нелинейного regressor.

    Используется в `MyModel.count_parameters()` для BIC и подобных оценок.
    Для tree-ensembles считаем общее число листьев (каждый лист ~ одно "параметр"-предсказание).
    Для KNN используем n_neighbors как эффективное сглаживание (DOF ≈ n/k).
    Fallback — r + 1 (как линейная без свободного члена).

    Parameters
    ----------
    regressor : fitted estimator
    r : int
        Размер conditioning set (используется в fallback)

    Returns
    -------
    int
        Оценка числа параметров
    """
    # sklearn RandomForest / ExtraTrees / GradientBoosting
    if hasattr(regressor, "estimators_"):
        estimators = regressor.estimators_
        total_leaves = 0
        try:
            # GradientBoostingRegressor: estimators_ shape (n_estimators, 1) of DecisionTreeRegressor
            # RandomForestRegressor: estimators_ is list of DecisionTreeRegressor
            flat = np.ravel(estimators)
            for est in flat:
                if hasattr(est, "tree_") and hasattr(est.tree_, "n_leaves"):
                    total_leaves += est.tree_.n_leaves
                else:
                    total_leaves += 1
            if total_leaves > 0:
                return int(total_leaves)
        except Exception:
            pass
    # XGBoost
    if hasattr(regressor, "get_booster"):
        try:
            booster = regressor.get_booster()
            dump = booster.get_dump()
            # Count leaves: each line "leaf=" is one leaf
            total_leaves = sum(tree.count("leaf=") for tree in dump)
            if total_leaves > 0:
                return int(total_leaves)
        except Exception:
            pass
    # KNN: эффективное DOF ≈ n_samples_fit / n_neighbors (но n_samples_fit ~ n, не имеем доступа)
    if hasattr(regressor, "n_neighbors"):
        # Просто возвращаем разумный fallback
        return int(regressor.n_neighbors) + r
    # Fallback — как линейная (r weights + intercept)
    return r + 1


class MyModel:
    """
    Модель разложения полной плотности в ряд с параметрическими условными гауссианами.

    На шаге k учим все плотности вида p(x_i | x_I), где:
    - x_i — одна переменная (целевая)
    - x_I — подмножество размера от 0 до k (обусловливающие переменные)
    - i не принадлежит I

    Каждая условная плотность параметризуется как:
    p(x_i | x_I) = N(mu_i + w_i^T x_I, sigma_i^2)

    Параметры (mu_i, w_i, sigma_i) обучаются линейной регрессией.

    Формула разложения:
    log p(x) = sum_{r=0}^{k} (1 / ((r+1) * C_d^{r+1})) * F^{(r)}

    где F^{(r)} = sum_{I: |I|=r} sum_{i not in I} log p(x_i | x_I)

    Parameters
    ----------
    d : int
        Размерность данных
    k : int
        Максимальный порядок обусловливания (количество переменных в условии)
        k=0: только маржинальные одномерные плотности
        k=1: плотности вида p(x_i) и p(x_i | x_j)
        k=d-1: полная авторегрессия
    regression_class : class
        Класс регрессии для обучения условных распределений (по умолчанию LinearRegression)
    regr_class_params : dict
        Параметры для класса регрессии
    """

    def __init__(self, d, k, regression_class=None, regr_class_params=None):
        from sklearn.linear_model import LinearRegression
        
        self.d = d
        self.k = min(k, d - 1)

        # Хранилище для параметров условных распределений
        # self.conditional_models[(i, I)] = {'mu': float, 'w': array, 'sigma': float}
        self.conditional_models = {}
        self.is_fitted = False
        self.n_densities_fitted = 0
        self.regression_class = regression_class if regression_class else LinearRegression
        self.regr_class_params = regr_class_params if regr_class_params else {}

    def fit(self, X):
        """
        Обучить все одномерные условные плотности.

        Процедура:
        1. Для каждого порядка r от 0 до k:
           - Для каждого подмножества I размера r (обусловливающие переменные)
             - Для каждой переменной i не в I (целевая переменная):
               - Обучить линейную регрессию x_i ~ x_I
               - Оценить дисперсию шума

        Parameters
        ----------
        X : array-like, shape (n, d)
            Обучающие данные

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        self.conditional_models = {}
        total_densities = 0

        # Итерируем по порядкам (размер обусловливающего множества)
        for r in range(self.k + 1):
            # Для каждого подмножества размера r
            for conditioning_subset in combinations(range(self.d), r):
                # Для каждой переменной, не в этом подмножестве
                for target_var in range(self.d):
                    if target_var not in conditioning_subset:
                        # Подготавливаем данные
                        y = X[:, target_var]  # целевая переменная

                        if len(conditioning_subset) == 0:
                            # Маржинальное распределение: p(x_i)
                            mu = np.mean(y)
                            w = np.array([])
                            sigma = max(float(np.std(y)), 1e-6)
                            regressor = None
                        else:
                            # Условное распределение: p(x_i | x_I)
                            X_conditioning = X[:, list(conditioning_subset)]

                            regressor = self.regression_class(**self.regr_class_params)
                            regressor.fit(X_conditioning, y)

                            # Для линейных регрессоров сохраняем (mu, w) — быстрый путь.
                            # Для нелинейных (RF/GB/XGBoost/KNN) — сохраняем сам regressor
                            # и используем predict() при вычислении log_prob.
                            if hasattr(regressor, "intercept_") and hasattr(regressor, "coef_"):
                                mu = regressor.intercept_
                                w = regressor.coef_
                                regressor = None  # не нужно хранить, mu/w достаточно
                            else:
                                mu = None
                                w = None

                            # Оцениваем дисперсию из остатков
                            y_pred = (regressor.predict(X_conditioning)
                                      if regressor is not None
                                      else mu + X_conditioning @ w)
                            residuals = y - y_pred
                            sigma = max(float(np.sqrt(np.mean(residuals**2))), 1e-6)

                        # Сохраняем параметры
                        key = (target_var, conditioning_subset)
                        self.conditional_models[key] = {
                            "mu": mu,
                            "w": w,
                            "sigma": sigma,
                            "regressor": regressor,  # None для линейных, else fitted regressor
                        }

                        total_densities += 1

        self.is_fitted = True
        self.n_densities_fitted = total_densities
        return self

    def fit_parallel(self, X, n_jobs=-1):
        """
        Обучить все одномерные условные плотности параллельно.

        Использует joblib для параллельного обучения условных распределений.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Обучающие данные
        n_jobs : int
            Количество параллельных процессов (-1 = все доступные ядра)

        Returns
        -------
        self
        """
        from joblib import Parallel, delayed
        
        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        # Собираем все задачи для параллельного выполнения
        tasks = []
        for r in range(self.k + 1):
            for conditioning_subset in combinations(range(self.d), r):
                for target_var in range(self.d):
                    if target_var not in conditioning_subset:
                        tasks.append((target_var, conditioning_subset))

        def _fit_single_density(target_var, conditioning_subset, X, regression_class, regr_class_params):
            """Обучить одну условную плотность."""
            y = X[:, target_var]

            if len(conditioning_subset) == 0:
                # Маржинальное распределение: p(x_i)
                mu = np.mean(y)
                w = np.array([])
                sigma = max(float(np.std(y)), 1e-6)
                regressor = None
            else:
                # Условное распределение: p(x_i | x_I)
                X_conditioning = X[:, list(conditioning_subset)]

                regressor = regression_class(**regr_class_params)
                regressor.fit(X_conditioning, y)

                if hasattr(regressor, "intercept_") and hasattr(regressor, "coef_"):
                    mu = regressor.intercept_
                    w = regressor.coef_
                    regressor = None
                else:
                    mu = None
                    w = None

                # Оцениваем дисперсию из остатков
                y_pred = (regressor.predict(X_conditioning)
                          if regressor is not None
                          else mu + X_conditioning @ w)
                residuals = y - y_pred
                sigma = max(float(np.sqrt(np.mean(residuals**2))), 1e-6)

            return (target_var, conditioning_subset), {
                "mu": mu, "w": w, "sigma": sigma, "regressor": regressor,
            }

        # Параллельное обучение
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_density)(
                target_var, conditioning_subset, X, 
                self.regression_class, self.regr_class_params
            )
            for target_var, conditioning_subset in tasks
        )

        # Собираем результаты
        self.conditional_models = {}
        for key, params in results:
            self.conditional_models[key] = params

        self.is_fitted = True
        self.n_densities_fitted = len(results)
        return self

    def _log_conditional_density(
        self, X_target, X_conditioning, target_var, conditioning_subset
    ):
        """
        Вычислить log условной плотности p(x_i | x_I).

        log p(x_i | x_I) = -0.5 * log(2π) - log(sigma_i) - 0.5 * ((x_i - (mu_i + w_i^T x_I)) / sigma_i)^2
        """
        key = (target_var, conditioning_subset)
        params = self.conditional_models[key]

        mu = params["mu"]
        w = params["w"]
        sigma = params["sigma"]
        regressor = params.get("regressor", None)

        if regressor is not None:
            # Нелинейный regressor (RF, GB, XGBoost, KNN и т.п.)
            mu_cond = regressor.predict(X_conditioning)
        elif w is None or len(w) == 0:
            # Маргинальная плотность (conditioning пустой)
            mu_cond = mu
        else:
            # Линейная регрессия: mu + w^T x
            mu_cond = mu + np.dot(X_conditioning, w)

        log_p = (
            -0.5 * np.log(2 * np.pi)
            - np.log(sigma)
            - 0.5 * ((X_target - mu_cond) / sigma) ** 2
        )

        return log_p

    def log_prob(self, X):
        """
        Вычислить log-плотность с использованием биномиальных коэффициентов.

        Формула:
        log p(x) = sum_{r=0}^{k} (1 / ((r+1) * C_d^{r+1})) * log F^{(r)}

        где:
        - F^{(r)} = sum_{I: |I|=r} sum_{i not in I} log p(x_i | x_I)
        - C_d^{r+1} = d! / ((r+1)! * (d-r-1)!) = "d выбрать r+1"
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X)")

        X = np.asarray(X)
        n, d = X.shape
        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        log_p_total = np.zeros(n)

        # Суммируем вклады по КАЖДОМУ ПОРЯДКУ отдельно
        for r in range(self.k + 1):
            # Суммируем все плотности ТОЛЬКО порядка r в F^{(r)}
            log_p_order_r = np.zeros(n)

            for (target_var, conditioning_subset), _ in self.conditional_models.items():
                # Берем только плотности порядка r
                if len(conditioning_subset) != r:
                    continue

                X_target = X[:, target_var]

                if len(conditioning_subset) > 0:
                    X_conditioning = X[:, conditioning_subset]
                else:
                    X_conditioning = np.array([]).reshape(n, 0)

                log_p_cond = self._log_conditional_density(
                    X_target, X_conditioning, target_var, conditioning_subset
                )

                log_p_order_r += log_p_cond

            # КЛЮЧЕВОЙ МОМЕНТ: коэффициент = 1 / ((r+1) * C_d^{r+1})
            if r != self.k:
                C_d_r_plus_1 = comb(self.d, r + 1)
                coef = 1 / ((r + 1) * C_d_r_plus_1)
            else:
                C_d_r_plus_1 = comb(self.d, r)
                coef = 1 / C_d_r_plus_1

            log_p_total += coef * log_p_order_r

        return log_p_total

    def count_parameters(self):
        """
        Подсчитать общее количество параметров в модели.

        Для каждой условной плотности:
        - Маргинальная (r=0): 2 (mu, sigma)
        - Линейная (saved via mu/w): r + 2 (intercept + r weights + sigma)
        - Нелинейная (saved regressor): попытка оценить через regressor-specific attrs;
          fallback — r + 2 как для линейной (эффективные DOF)
        """
        total_params = 0
        for (i, conditioning_subset), params in self.conditional_models.items():
            r = len(conditioning_subset)
            if r == 0:
                total_params += 2
                continue

            regressor = params.get("regressor", None)
            if regressor is None:
                # Линейная регрессия: r + 2
                total_params += r + 2
            else:
                # Нелинейный: попытка оценить реальное число параметров
                n_params = _estimate_regressor_params(regressor, r)
                total_params += n_params + 1  # + sigma
        return total_params

    def get_avg_sigma_squared(self):
        """
        Получить среднюю дисперсию sigma^2 по всем условным плотностям.
        Используется для оценки устойчивости параметров.
        """
        sigmas = []
        for params in self.conditional_models.values():
            sigmas.append(params["sigma"] ** 2)
        return np.mean(sigmas) if sigmas else 0.0

    def get_density_count(self):
        """Получить количество обученных плотностей по порядкам"""
        counts = {}
        for r in range(self.k + 1):
            count = sum(
                1
                for (_, conditioning_subset), _ in self.conditional_models.items()
                if len(conditioning_subset) == r
            )
            counts[r] = count
        return counts

    def get_feature_matrix_old(self, X):
        """
        Вычислить матрицу признаков для каждого объекта.

        Для каждого объекта вычисляем матрицу [d, k+1], где элемент s(i, r) равен:
        
        s(i, r) = sum_{I in I_r; i not in I} log p(x_i | x_I)
        
        Это сумма log-правдоподобий для переменной i по всем обусловливающим 
        подмножествам размера r, не содержащим i.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Данные для вычисления признаков

        Returns
        -------
        feature_matrix : ndarray, shape (n, d, k+1)
            Матрица признаков: feature_matrix[obj, i, r] = s(i, r) для объекта obj

            Проблема: s(i, 0) --> s(i, 1) --> ... s(i, n) --- т.е. они являются последовательными приближениями одного и того же числа s(i) --- 
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        # Матрица признаков: [n объектов, d переменных, k+1 порядков]
        feature_matrix = np.zeros((n, self.d, self.k + 1))

        # Итерируем по всем обученным условным плотностям
        for (target_var, conditioning_subset), params in self.conditional_models.items():
            r = len(conditioning_subset)  # порядок (размер условия) = размер ограничения

            # Вычисляем log p(x_i | x_I) для всех объектов
            X_target = X[:, target_var] # i-ая координата у всех объектов

            if r > 0:
                X_conditioning = X[:, list(conditioning_subset)] # обуславливаемся
            else:
                X_conditioning = np.array([]).reshape(n, 0)

            log_p_cond = self._log_conditional_density(
                X_target, X_conditioning, target_var, conditioning_subset
            )

            # Добавляем к соответствующему элементу матрицы
            # s(i, r) += log p(x_i | x_I)
            feature_matrix[:, target_var, r] += log_p_cond

        
        return feature_matrix
    

    def get_feature_matrix(self, X):
        """
        Вычислить матрицу признаков для каждого объекта.

        Для каждого объекта вычисляем матрицу [d, k+1], где элемент s(i, r) равен:
        
        s(i, r) = sum_{I in I_r; i not in I} log p(x_i | x_I)
        
        Это сумма log-правдоподобий для переменной i по всем обусловливающим 
        подмножествам размера r, не содержащим i.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Данные для вычисления признаков

        Returns
        -------
        feature_matrix : ndarray, shape (n, d, k+1)
            Матрица признаков: feature_matrix[obj, i, r] = s(i, r) для объекта obj

        НЕТ: сейчас уже (n, n_features)


            Проблема: s(i, 0) --> s(i, 1) --> ... s(i, n) --- т.е. они являются последовательными приближениями одного и того же числа s(i) --- 
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        # Матрица признаков: [n объектов, d переменных, k+1 порядков]
        feature_matrix = np.zeros((n, self.d, self.k + 1))

        # Итерируем по всем обученным условным плотностям
        for (target_var, conditioning_subset), params in self.conditional_models.items():
            r = len(conditioning_subset)  # порядок (размер условия) = размер ограничения

            # Вычисляем log p(x_i | x_I) для всех объектов
            X_target = X[:, target_var] # i-ая координата у всех объектов

            if r > 0:
                X_conditioning = X[:, list(conditioning_subset)] # обуславливаемся
            else:
                X_conditioning = np.array([]).reshape(n, 0)

            log_p_cond = self._log_conditional_density(
                X_target, X_conditioning, target_var, conditioning_subset
            )

            # Добавляем к соответствующему элементу матрицы
            # s(i, r) += log p(x_i | x_I)
            feature_matrix[:, target_var, r] += log_p_cond


        n = feature_matrix.shape[0]
    
        features = feature_matrix.reshape(n, -1)
        
        return features, [""] * features.shape[1]


    def get_feature_matrix_full(self, X):
        """
        Вычислить полную матрицу признаков — все индивидуальные log-условные плотности.

        Для каждого объекта возвращаем вектор длины
            sum_{r=0}^{k} (r+1) * C(d, r+1)
        
        где каждый элемент — это log p(x_i | x_I) для конкретной пары (i, I).

        Признаки:
            k=0:  log p(x_i)                   — d штук
            k=1:  log p(x_i | x_j)             — d*(d-1) штук
            k=2:  log p(x_i | x_j, x_k)        — 3*C(d,3) штук
            ...
            k=r:  (r+1)*C(d, r+1) штук

        Parameters
        ----------
        X : array-like, shape (n, d)
            Данные для вычисления признаков

        Returns
        -------
        feature_matrix : ndarray, shape (n, total_features)
            Полная матрица признаков
        feature_names : list of str
            Имена признаков вида "logp(x_i|x_I)" для каждого столбца
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        # Собираем все признаки в определённом порядке: r=0, r=1, ..., r=k
        columns = []     # список массивов shape (n,)
        feature_names = []

        for r in range(self.k + 1):
            for conditioning_subset in combinations(range(self.d), r):
                for target_var in range(self.d):
                    if target_var not in conditioning_subset:
                        key = (target_var, conditioning_subset)
                        if key not in self.conditional_models:
                            continue

                        X_target = X[:, target_var]
                        if r > 0:
                            X_conditioning = X[:, list(conditioning_subset)]
                        else:
                            X_conditioning = np.array([]).reshape(n, 0)

                        log_p_cond = self._log_conditional_density(
                            X_target, X_conditioning, target_var, conditioning_subset
                        )

                        columns.append(log_p_cond)

                        # Формируем имя признака
                        if r == 0:
                            fname = f"logp(x_{target_var})"
                        else:
                            cond_str = ",".join(str(c) for c in conditioning_subset)
                            fname = f"logp(x_{target_var}|x_{{{cond_str}}})"
                        feature_names.append(fname)

        feature_matrix = np.column_stack(columns) if columns else np.zeros((n, 0))
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        return feature_matrix, feature_names


    def get_feature_matrix_full_aggregated(self, X):
        """
        Вычислить агрегированную матрицу признаков с описательными статистиками.

        Для каждого порядка r и каждой целевой переменной i собираем набор значений:
            {log p(x_i | x_I) : |I| = r, i ∉ I}

        Затем:
        - Для r = 0 и r = 1: **сырые признаки** + описательные статистики 
          (mean, median, std, min, max) по всем условиям для каждой переменной i
        - Для r >= 2: **только** описательные статистики (mean, median, std, min, max)
          для каждой переменной i

        Это позволяет снизить экспоненциальную сложность для высоких порядков,
        сохраняя при этом информативность.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Данные для вычисления признаков

        Returns
        -------
        feature_matrix : ndarray, shape (n, total_features)
            Агрегированная матрица признаков
        feature_names : list of str
            Имена признаков для каждого столбца
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected data with {self.d} dimensions, got {d}")

        stat_names = ["mean", "median", "std", "min", "max"]

        columns = []
        feature_names = []

        for r in range(self.k + 1):
            # Для каждой целевой переменной i собираем все log p(x_i | x_I) при |I|=r
            per_variable = {}  # target_var -> list of (conditioning_subset, log_p array)

            for conditioning_subset in combinations(range(self.d), r):
                for target_var in range(self.d):
                    if target_var not in conditioning_subset:
                        key = (target_var, conditioning_subset)
                        if key not in self.conditional_models:
                            continue

                        X_target = X[:, target_var]
                        if r > 0:
                            X_conditioning = X[:, list(conditioning_subset)]
                        else:
                            X_conditioning = np.array([]).reshape(n, 0)

                        log_p_cond = self._log_conditional_density(
                            X_target, X_conditioning, target_var, conditioning_subset
                        )

                        if target_var not in per_variable:
                            per_variable[target_var] = []
                        per_variable[target_var].append((conditioning_subset, log_p_cond))

            # Теперь формируем признаки
            for target_var in sorted(per_variable.keys()):
                entries = per_variable[target_var]

                # --- Сырые признаки (только для r <= 1) ---
                if r <= 1:
                    for conditioning_subset, log_p_cond in entries:
                        columns.append(log_p_cond)
                        if r == 0:
                            fname = f"logp(x_{target_var})"
                        else:
                            cond_str = ",".join(str(c) for c in conditioning_subset)
                            fname = f"logp(x_{target_var}|x_{{{cond_str}}})"
                        feature_names.append(fname)

                # --- Описательные статистики ---
                # Стекаем все log p(x_i | x_I) для данного (i, r) в матрицу (n, num_subsets)
                all_log_p = np.column_stack([lp for _, lp in entries])  # shape (n, num_subsets)

                stats_mean = np.mean(all_log_p, axis=1)
                stats_median = np.median(all_log_p, axis=1)
                stats_std = np.std(all_log_p, axis=1)
                stats_min = np.min(all_log_p, axis=1)
                stats_max = np.max(all_log_p, axis=1)

                for stat_val, stat_name in zip(
                    [stats_mean, stats_median, stats_std, stats_min, stats_max],
                    stat_names
                ):
                    columns.append(stat_val)
                    feature_names.append(f"r{r}_x{target_var}_{stat_name}")

        feature_matrix = np.column_stack(columns) if columns else np.zeros((n, 0))
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        return feature_matrix, feature_names

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"MyModel(d={self.d}, k={self.k}, {status}, n_densities={len(self.conditional_models)})"



"""Теперь помио функции get_feature_matrix реализуй еще 2 функции. Вот описание идеи в latex: ```    \item В паренклитике производить моделирование высшего порядка (гиперребер). Соответствует шагу 2 и выше. Условие: достаточный объем данных, чтобы добавки $F^{(k)}$ оказали ощутимое влияние.
    \item В паренклитике считать ПОЛНОЕ правдоподобие данных (включая по-вершинные, скалярные правдоподобия).
    \item Полные фичи для модели --- логарифмы условных плотностей вершин. Их количество растет экспоненциально при увеличении сложности модели. Т.е.:

    \begin{equation}
        \begin{tab}
            k=0:& \log p(x_i) & 1 \cdot C_{d}^{1}\\
            k=1 (\text{паренклитика}):& \log p(x_i|x_j)& 2 \cdot C_{d}^{2}\\
            k=2:& \log p(x_i|x_j, x_k) & 3 \cdot C_{d}^{3}\\
            \dots\\
            k=d-1:& \log p(x_i|x_1, \dots, x_d) & d \cdot C_{d}^{d}\\
        \end{cases}
    \end{equation}

    Для снижения экспоненциальной сложности, можно по каждому набору извлечь описательные статистики (как делали авторы статьи про паренклитику для частного случая $k=1$).

    Также можно для каждой мощности просуммировать условные плотности по всем наборам условий. Тогда для каждой вершины получим ее "силу". В случае с $k=1$ авторы статьи про паренклитику доказали, что этот признак самый хороший.
    Фичи текущей модели --- это суммы log-плотностей для каждой вершины каждой мощности, т.е. числа:

        $$
            s(i, k) = \sum\limits_{I \in \mathcal{I}_k; i \not \in I}\log p(x_i|x_I)
        $$
```

get_feature_matrix_full --- должна вычислять все признаки от условия размера 0 до условия размера k

get_feature_matrix_full_aggregated --- должна вычислять все признаки от условия размера 0 до условия размера k и затем для признаков с k >=2 (размера условия) использовать вместо сырых описательные статистики: среднее, медиану, std, min, max. Для признаков k<=1 эти описательные статистики тоже должны вычисляться и использоваться вместе с исходными признаками"""