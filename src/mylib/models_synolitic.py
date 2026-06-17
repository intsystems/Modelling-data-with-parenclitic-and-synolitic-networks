"""
Модель MyModelSynolitic — k-порядковое приближение условной плотности p(y|x)
через формулу обращения Мёбиуса.

Теоретическая основа (из статьи):
    log p(y|x) ≈_k Σ_{t=0}^{k} c^(k)(t,d) * Σ_{T⊂[d];|T|=t} log p(y|x_T)
                    + log p(y) * A_{d,k} + S^(k)(x)

где:
    c^(k)(t, d) = Σ_{s=0}^{k-t} (-1)^s * C(d-t, s)
    A_{d,k}     = 1 - Σ_{t=0}^{k} c^(k)(t, d) * C(d, t)
    S^(k)(x)    — нормирующий множитель (для классификации не нужен)

Частные случаи:
    k=0:  log p(y|x) ≈ log p(y)
    k=1:  log p(y|x) ≈ (1-d)*log p(y) + Σ_i log p(y|x_i)       (Naive Bayes через синолитику)
    k=2:  log p(y|x) ≈ C(d-1,2)*log p(y) + (2-d)*Σ_i log p(y|x_i) + Σ_{i<j} log p(y|x_i,x_j)
"""

import numpy as np
from itertools import combinations
from math import comb


class MyModelSynolitic:
    """
    Классификационная модель на основе k-порядкового приближения Мёбиуса.

    Для каждого подмножества признаков T размера t <= k обучается классификатор,
    моделирующий p(y | x_T). Затем предсказание собирается через взвешенную
    сумму с коэффициентами Мёбиуса:

        log p(y|x) ≈_k  Σ_{t=0}^{k} c^(k)(t,d) * Σ_{T:|T|=t} log p(y|x_T)
                        + A_{d,k} * log p(y)

    Признаки (feature matrix):
        s(i, t) = Σ_{T∋i, |T|=t} log p(y=1 | x_T)
        — «сила» вершины i на уровне t

    Parameters
    ----------
    d : int
        Размерность пространства признаков
    k : int
        Максимальный порядок подмножества условия (0 <= k <= d)
        k=0: только приор p(y)
        k=1: добавляются одномерные классификаторы (аналог Naive Bayes)
        k=2: добавляются попарные классификаторы (синолитика)
    classifier_class : class, optional
        Класс классификатора, умеющий predict_proba / predict_log_proba.
        По умолчанию LogisticRegression.
    clf_class_params : dict, optional
        Параметры для инициализации классификатора.
    """

    def __init__(self, d, k, classifier_class=None, clf_class_params=None):
        from sklearn.linear_model import LogisticRegression

        self.d = d
        self.k = min(k, d)

        self.classifier_class = (
            classifier_class if classifier_class is not None else LogisticRegression
        )
        self.clf_class_params = (
            clf_class_params if clf_class_params is not None else {"max_iter": 1000}
        )

        # Хранилище моделей: tuple(subset) -> fitted classifier (или None для t=0)
        self.conditional_models = {}

        # Приоры классов: class_label -> log p(y=class_label)
        self.log_prior = {}
        self.classes_ = None

        self.is_fitted = False
        self.n_classifiers_fitted = 0

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _mobius_coeff(self, t: int) -> float:
        """
        Коэффициент Мёбиуса c^(k)(t, d).

            c^(k)(t, d) = Σ_{s=0}^{k-t} (-1)^s * C(d-t, s)

        При t > k возвращает 0 (по определению k-усечения).
        """
        if t > self.k:
            return 0.0
        result = 0.0
        for s in range(self.k - t + 1):
            result += ((-1) ** s) * comb(self.d - t, s)
        return result

    def _A_coeff(self) -> float:
        """
        Поправочный коэффициент для приора:
            A_{d,k} = 1 - Σ_{t=0}^{k} c^(k)(t, d) * C(d, t)
        """
        total = sum(
            self._mobius_coeff(t) * comb(self.d, t) for t in range(self.k + 1)
        )
        return 1.0 - total

    def _log_prob_subset(
        self, X: np.ndarray, subset: tuple, target_class
    ) -> np.ndarray:
        """
        Вычислить log p(y=target_class | x_T) для каждого объекта.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        subset : tuple — индексы признаков подмножества T
        target_class : метка класса

        Returns
        -------
        log_p : ndarray, shape (n,)
        """
        n = X.shape[0]
        t = len(subset)

        if t == 0:
            # Только приор: log p(y=target_class)
            log_p_val = self.log_prior.get(target_class, -np.inf)
            return np.full(n, log_p_val)

        clf = self.conditional_models[subset]
        X_sub = X[:, list(subset)]
        log_proba = clf.predict_log_proba(X_sub)  # (n, n_classes)

        class_idx = list(clf.classes_).index(target_class)
        return log_proba[:, class_idx]

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучить классификаторы p(y | x_T) для всех подмножеств T размера 0..k.

        Для t=0: сохраняем только эмпирические приоры классов.
        Для t=1..k: обучаем classifier_class на признаках x_T.

        Parameters
        ----------
        X : array-like, shape (n, d)
        y : array-like, shape (n,) — метки классов

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected {self.d} features, got {d}")

        self.classes_ = np.unique(y)

        # Вычисляем log-приоры
        self.log_prior = {}
        for c in self.classes_:
            p_c = np.mean(y == c)
            self.log_prior[c] = np.log(p_c + 1e-15)

        self.conditional_models = {}
        total_classifiers = 0

        for t in range(0, self.k + 1):
            for subset in combinations(range(d), t):
                if t == 0:
                    # t=0: только приор, отдельная модель не нужна
                    self.conditional_models[subset] = None
                else:
                    X_sub = X[:, list(subset)]
                    clf = self.classifier_class(**self.clf_class_params)
                    clf.fit(X_sub, y)
                    self.conditional_models[subset] = clf
                    total_classifiers += 1

        self.is_fitted = True
        self.n_classifiers_fitted = total_classifiers
        return self

    def fit_parallel(self, X: np.ndarray, y: np.ndarray, n_jobs: int = -1):
        """
        Параллельное обучение через joblib.

        Parameters
        ----------
        X : array-like, shape (n, d)
        y : array-like, shape (n,)
        n_jobs : int — количество параллельных процессов (-1 = все ядра)

        Returns
        -------
        self
        """
        from joblib import Parallel, delayed

        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected {self.d} features, got {d}")

        self.classes_ = np.unique(y)
        self.log_prior = {
            c: np.log(np.mean(y == c) + 1e-15) for c in self.classes_
        }

        # Задачи только для t >= 1
        tasks = [
            subset
            for t in range(1, self.k + 1)
            for subset in combinations(range(d), t)
        ]

        def _fit_one(subset, X, y, clf_class, clf_params):
            X_sub = X[:, list(subset)]
            clf = clf_class(**clf_params)
            clf.fit(X_sub, y)
            return subset, clf

        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(
                subset, X, y, self.classifier_class, self.clf_class_params
            )
            for subset in tasks
        )

        self.conditional_models = {(): None}  # t=0: только приор
        for subset, clf in results:
            self.conditional_models[subset] = clf

        self.is_fitted = True
        self.n_classifiers_fitted = len(results)
        return self

    # ------------------------------------------------------------------
    # Предсказания
    # ------------------------------------------------------------------

    def log_prob(self, X: np.ndarray, target_class=1) -> np.ndarray:
        """
        Приближённый log p(y=target_class | x) по формуле Мёбиуса.

            log p(y|x) ≈_k  Σ_{t=0}^{k} c^(k)(t,d) * Σ_{T:|T|=t} log p(y|x_T)
                            + A_{d,k} * log p(y)

        Parameters
        ----------
        X : array-like, shape (n, d)
        target_class : метка класса (по умолчанию 1)

        Returns
        -------
        log_p : ndarray, shape (n,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X, y)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected {self.d} features, got {d}")

        log_p = np.zeros(n)

        for t in range(self.k + 1):
            c = self._mobius_coeff(t)
            if c == 0.0:
                continue
            for subset in combinations(range(d), t):
                log_p += c * self._log_prob_subset(X, subset, target_class)

        # Поправка на приор: A_{d,k} * log p(y)
        A = self._A_coeff()
        log_p += A * self.log_prior.get(target_class, -np.inf)

        return log_p

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Вычислить log p(y=c | x) для каждого класса c из classes_.

        Для каждого класса вычисляется Мёбиус-приближение log p(y=c|x),
        затем результат нормируется через log-sum-exp.

        Parameters
        ----------
        X : array-like, shape (n, d)

        Returns
        -------
        log_proba : ndarray, shape (n, n_classes)
            Нормированные log-вероятности; порядок столбцов совпадает с classes_.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X, y)")

        X = np.asarray(X)

        raw = np.column_stack(
            [self.log_prob(X, target_class=c) for c in self.classes_]
        )  # (n, n_classes)

        # Стабильная нормировка через log-sum-exp
        row_max = raw.max(axis=1, keepdims=True)
        log_sum = row_max + np.log(np.exp(raw - row_max).sum(axis=1, keepdims=True))
        return raw - log_sum

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Вычислить p(y=c | x) для каждого класса (нормированные вероятности).

        Parameters
        ----------
        X : array-like, shape (n, d)

        Returns
        -------
        proba : ndarray, shape (n, n_classes)
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать класс для каждого объекта (argmax по классам).

        Parameters
        ----------
        X : array-like, shape (n, d)

        Returns
        -------
        y_pred : ndarray, shape (n,)
        """
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    # ------------------------------------------------------------------
    # Матрицы признаков
    # ------------------------------------------------------------------

    def get_feature_matrix(self, X: np.ndarray, target_class=1) -> np.ndarray:
        """
        Вычислить матрицу признаков «сила вершины».

        Для каждого объекта, переменной i и порядка t:

            s(i, t) = Σ_{T∋i, |T|=t} log p(y=target_class | x_T)

        Сумма log-правдоподобий по всем подмножествам размера t, содержащим i.
        При t=0: s(i, 0) = log p(y) — одинаково для всех i.

        Parameters
        ----------
        X : array-like, shape (n, d)
        target_class : метка класса (по умолчанию 1)

        Returns
        -------
        feature_matrix : ndarray, shape (n, d * (k+1))
            Развёрнутый тензор (n, d, k+1):
            feature_matrix[obj, i*(k+1) + t] = s(i, t)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X, y)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected {self.d} features, got {d}")

        feat_3d = np.zeros((n, d, self.k + 1))

        # t=0: приор — одинаков для всех вершин
        feat_3d[:, :, 0] = self.log_prior.get(target_class, -np.inf)

        # t=1..k: суммируем по подмножествам, содержащим вершину i
        for t in range(1, self.k + 1):
            for subset in combinations(range(d), t):
                log_p = self._log_prob_subset(X, subset, target_class)  # (n,)
                for i in subset:
                    feat_3d[:, i, t] += log_p

        return feat_3d.reshape(n, -1)

    def get_feature_matrix_full(self, X: np.ndarray, target_class=1):
        """
        Полная матрица признаков — все индивидуальные log p(y | x_T).

        Для каждого подмножества T размера t (0 <= t <= k) — отдельный признак.
        Количество признаков: Σ_{t=0}^{k} C(d, t)

            t=0: log p(y=1)                         — 1 признак
            t=1: log p(y=1 | x_i), i=0..d-1        — d признаков
            t=2: log p(y=1 | x_i, x_j), i<j        — C(d,2) признаков
            ...

        Parameters
        ----------
        X : array-like, shape (n, d)
        target_class : метка класса (по умолчанию 1)

        Returns
        -------
        feature_matrix : ndarray, shape (n, Σ C(d,t))
        feature_names : list of str
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X, y)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected {self.d} features, got {d}")

        columns = []
        feature_names = []

        for t in range(self.k + 1):
            for subset in combinations(range(d), t):
                log_p = self._log_prob_subset(X, subset, target_class)
                columns.append(log_p)

                if t == 0:
                    feature_names.append(f"logp(y={target_class})")
                else:
                    feat_str = ",".join(str(i) for i in subset)
                    feature_names.append(f"logp(y={target_class}|x_{{{feat_str}}})")

        feature_matrix = np.column_stack(columns) if columns else np.zeros((n, 0))
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        return feature_matrix, feature_names

    def get_feature_matrix_full_aggregated(self, X: np.ndarray, target_class=1):
        """
        Агрегированная матрица признаков с описательными статистиками.

        Для каждого порядка t и каждой переменной i:
            { log p(y | x_T) : |T| = t, i ∈ T }  — набор значений

        Политика:
        - t=0: скалярный приор (1 признак)
        - t=1: сырой признак log p(y|x_i) + 5 статистик (mean/median/std/min/max)
        - t>=2: только 5 статистик на вершину (сырые признаки отбрасываются
                для снижения экспоненциальной сложности)

        Parameters
        ----------
        X : array-like, shape (n, d)
        target_class : метка класса (по умолчанию 1)

        Returns
        -------
        feature_matrix : ndarray, shape (n, total_features)
        feature_names : list of str
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(X, y)")

        X = np.asarray(X)
        n, d = X.shape

        if d != self.d:
            raise ValueError(f"Expected {self.d} features, got {d}")

        stat_fns = [np.mean, np.median, np.std, np.min, np.max]
        stat_names = ["mean", "median", "std", "min", "max"]

        columns = []
        feature_names = []

        # --- t=0: приор ---
        columns.append(np.full(n, self.log_prior.get(target_class, -np.inf)))
        feature_names.append(f"logp(y={target_class})")

        # --- t=1..k ---
        for t in range(1, self.k + 1):
            # per_variable[i] = список массивов log p(y | x_T) для T∋i, |T|=t
            per_variable: dict = {i: [] for i in range(d)}

            for subset in combinations(range(d), t):
                log_p = self._log_prob_subset(X, subset, target_class)  # (n,)
                for i in subset:
                    per_variable[i].append(log_p)

            for i in range(d):
                entries = per_variable[i]
                if not entries:
                    continue

                # Сырой признак — только для t=1 (ровно одно подмножество {i})
                if t == 1:
                    columns.append(entries[0])
                    feature_names.append(f"logp(y={target_class}|x_{i})")

                # Описательные статистики — для всех t>=1
                all_log_p = np.column_stack(entries)  # (n, num_subsets_containing_i)
                for fn, sname in zip(stat_fns, stat_names):
                    columns.append(fn(all_log_p, axis=1))
                    feature_names.append(f"t{t}_x{i}_{sname}")

        feature_matrix = np.column_stack(columns) if columns else np.zeros((n, 0))
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        return feature_matrix, feature_names

    # ------------------------------------------------------------------
    # Информационные / диагностические методы
    # ------------------------------------------------------------------

    def get_mobius_coefficients(self) -> dict:
        """
        Вернуть все коэффициенты Мёбиуса и поправочный коэффициент.

        Returns
        -------
        dict:
            'c' : list[float] — c^(k)(t, d) для t = 0..k
            'A' : float       — A_{d,k} (поправка к приору)
        """
        c_vals = [self._mobius_coeff(t) for t in range(self.k + 1)]
        return {"c": c_vals, "A": self._A_coeff()}

    def get_classifier_count(self) -> dict:
        """
        Количество обученных классификаторов по порядкам t.

        Returns
        -------
        dict: {t: count}
        """
        return {
            t: sum(1 for s in self.conditional_models if len(s) == t)
            for t in range(self.k + 1)
        }

    def count_parameters(self) -> int:
        """
        Суммарное число параметров всех классификаторов.

        Для LogisticRegression с t признаками: t коэффициентов + 1 intercept.
        """
        total = 0
        for subset, clf in self.conditional_models.items():
            t = len(subset)
            if t == 0 or clf is None:
                total += len(self.classes_) if self.classes_ is not None else 0
            elif hasattr(clf, "coef_"):
                total += clf.coef_.size + clf.intercept_.size
            else:
                total += t + 1
        return total

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        if self.is_fitted:
            coeffs = self.get_mobius_coefficients()
            c_str = ", ".join(
                f"c({t})={v:.3f}" for t, v in enumerate(coeffs["c"])
            )
            A_str = f", A={coeffs['A']:.3f}"
        else:
            c_str, A_str = "", ""
        return (
            f"MyModelSynolitic(d={self.d}, k={self.k}, {status}, "
            f"n_classifiers={self.n_classifiers_fitted}"
            + (f", [{c_str}{A_str}]" if c_str else "")
            + ")"
        )
