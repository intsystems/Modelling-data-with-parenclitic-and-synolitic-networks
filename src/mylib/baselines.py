"""
Baseline модели для сравнения с MyModel.

Классы:
- ARModel_Fixed: Авторегрессия с фиксированным порядком переменных
- ARModel_Random: Авторегрессия со случайной перестановкой переменных
- ARModel_Average: Усреднение по нескольким случайным авторегрессиям
- Gaussian_Full: Полное многомерное нормальное распределение
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal


class ARModel_Fixed:
    """
    Авторегрессия: p(x) = ∏_{i=1}^d p(x_i | x_{1},...,x_{i-1})
    Используется только ОДИН фиксированный порядок (например, 1,2,...,d)
    
    Parameters
    ----------
    d : int
        Размерность данных
    k : int
        Максимальный порядок зависимости (количество предыдущих переменных)
    """

    def __init__(self, d, k):
        self.d = d
        self.k = min(k, d - 1)
        self.models = {}  # {i: regressor_for_x_i}
        self.is_fitted = False

    def fit(self, X):
        """
        Обучить авторегрессионную модель.
        
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

        self.models = {}

        for i in range(self.d):
            # Кондиционируем на предыдущих k элементов в фиксированном порядке
            conditioning_indices = list(range(max(0, i - self.k), i))

            y = X[:, i]

            if len(conditioning_indices) == 0:
                # Маржинальное распределение
                mu = np.mean(y)
                sigma = np.std(y)
                self.models[i] = {"type": "marginal", "mu": mu, "sigma": sigma}
            else:
                # Регрессия x_i на conditioning_indices
                X_cond = X[:, conditioning_indices]
                regressor = LinearRegression()
                regressor.fit(X_cond, y)

                residuals = y - regressor.predict(X_cond)
                sigma = np.sqrt(np.mean(residuals**2))

                self.models[i] = {
                    "type": "conditional",
                    "intercept": regressor.intercept_,
                    "coef": regressor.coef_,
                    "sigma": sigma,
                    "conditioning_indices": conditioning_indices,
                }

        self.is_fitted = True
        return self

    def log_prob(self, X):
        """
        Вычислить log-вероятность для каждого сэмпла.
        
        Parameters
        ----------
        X : array-like, shape (n, d)
            Данные для оценки
        
        Returns
        -------
        log_p : array, shape (n,)
            Log-вероятности
        """
        X = np.asarray(X)
        n, d = X.shape

        log_p = np.zeros(n)

        for i in range(self.d):
            model_i = self.models[i]
            y = X[:, i]

            if model_i["type"] == "marginal":
                mu = model_i["mu"]
                sigma = model_i["sigma"]
                mu_cond = np.full(n, mu)
            else:
                X_cond = X[:, model_i["conditioning_indices"]]
                mu_cond = model_i["intercept"] + np.dot(X_cond, model_i["coef"])
                sigma = model_i["sigma"]

            log_p += (
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma)
                - 0.5 * ((y - mu_cond) / sigma) ** 2
            )

        return log_p

    def count_parameters(self):
        """Подсчитать количество параметров модели."""
        total = 0
        for i in range(self.d):
            model_i = self.models[i]
            if model_i["type"] == "marginal":
                total += 2  # mu, sigma
            else:
                total += len(model_i["coef"]) + 2  # coef, intercept, sigma
        return total

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ARModel_Fixed(d={self.d}, k={self.k}, {status})"


class ARModel_Random:
    """
    Авторегрессия со СЛУЧАЙНОЙ перестановкой переменных.
    Показывает, что порядок переменных влияет на качество.
    
    Parameters
    ----------
    d : int
        Размерность данных
    k : int
        Максимальный порядок зависимости
    random_state : int, optional
        Случайное состояние для перестановки
    """

    def __init__(self, d, k, random_state=None):
        self.d = d
        self.k = min(k, d - 1)
        self.random_state = random_state
        self.permutation = None
        self.ar_model = None
        self.is_fitted = False

    def fit(self, X):
        """
        Обучить модель со случайной перестановкой.
        
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

        # Случайная перестановка
        rng = np.random.RandomState(self.random_state)
        self.permutation = rng.permutation(self.d)

        # Переставляем данные
        X_permuted = X[:, self.permutation]

        # Обучаем ARModel_Fixed на переставленных данных
        self.ar_model = ARModel_Fixed(self.d, self.k)
        self.ar_model.fit(X_permuted)

        self.is_fitted = True
        return self

    def log_prob(self, X):
        """Вычислить log-вероятность."""
        X = np.asarray(X)
        # Переставляем данные
        X_permuted = X[:, self.permutation]
        return self.ar_model.log_prob(X_permuted)

    def count_parameters(self):
        """Подсчитать количество параметров."""
        return self.ar_model.count_parameters()

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ARModel_Random(d={self.d}, k={self.k}, {status})"


class ARModel_Average:
    """
    Усреднение log-вероятности по M случайным авторегрессионным порядкам.
    Показывает, что наивное усреднение по случайным порядкам
    не настолько хорошо, как осмысленная симметризация MyModel.
    
    Parameters
    ----------
    d : int
        Размерность данных
    k : int
        Максимальный порядок зависимости
    n_random_orders : int
        Количество случайных порядков для усреднения
    random_state : int, optional
        Случайное состояние
    """

    def __init__(self, d, k, n_random_orders=10, random_state=None):
        self.d = d
        self.k = min(k, d - 1)
        self.n_random_orders = n_random_orders
        self.random_state = random_state
        self.ar_models = []
        self.is_fitted = False

    def fit(self, X):
        """
        Обучить несколько моделей со случайными перестановками.
        
        Parameters
        ----------
        X : array-like, shape (n, d)
            Обучающие данные
        
        Returns
        -------
        self
        """
        X = np.asarray(X)

        rng = np.random.RandomState(self.random_state)
        self.ar_models = []
        
        for i in range(self.n_random_orders):
            ar = ARModel_Random(self.d, self.k, random_state=rng.randint(0, 100000))
            ar.fit(X)
            self.ar_models.append(ar)

        self.is_fitted = True
        return self

    def log_prob(self, X):
        """
        Вычислить усредненную log-вероятность.
        Усреднение производится в пространстве вероятностей (не логарифмов).
        """
        X = np.asarray(X)
        n = X.shape[0]

        # Усредняем в пространстве вероятностей
        probs = np.zeros(n)
        for ar_model in self.ar_models:
            probs += np.exp(ar_model.log_prob(X))
        probs /= self.n_random_orders

        return np.log(probs)

    def count_parameters(self):
        """Подсчитать количество параметров (сумма по всем моделям)."""
        return sum(ar.count_parameters() for ar in self.ar_models)

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ARModel_Average(d={self.d}, k={self.k}, n_orders={self.n_random_orders}, {status})"


class Gaussian_Full:
    """
    Полное многомерное нормальное распределение.
    Это верхняя граница качества для параметрических моделей,
    но требует O(d^2) параметров на ковариацию.
    
    Parameters
    ----------
    d : int
        Размерность данных
    """

    def __init__(self, d, **kwargs):
        self.d = d
        self.mu = None
        self.cov = None
        self.is_fitted = False

    def fit(self, X):
        """
        Обучить полное многомерное нормальное распределение.
        
        Parameters
        ----------
        X : array-like, shape (n, d)
            Обучающие данные
        
        Returns
        -------
        self
        """
        X = np.asarray(X)
        self.mu = X.mean(axis=0)
        self.cov = np.cov(X.T)

        # Регуляризация для численной стабильности
        self.cov += 1e-6 * np.eye(self.d)

        self.is_fitted = True
        return self

    def log_prob(self, X):
        """Вычислить log-вероятность."""
        X = np.asarray(X)
        dist = multivariate_normal(mean=self.mu, cov=self.cov)
        return dist.logpdf(X)

    def count_parameters(self):
        """
        Подсчитать количество параметров.
        mu: d параметров
        cov: d(d+1)/2 параметров (симметричная матрица)
        """
        return self.d + self.d * (self.d + 1) // 2

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"Gaussian_Full(d={self.d}, {status})"
