"""
Функции для проведения экспериментов с k-order зависимостями.

Функции:
- _run_single_seed: Запуск эксперимента для одного seed (вариация k)
- _run_single_seed_d_experiment: Запуск эксперимента для одного seed (вариация d)
- run_k_experiment_parallel: Параллельный эксперимент по перебору k
- run_d_experiment_parallel: Параллельный эксперимент по перебору d
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from models import KOrderDependencyGaussian, MyModel


def _run_single_seed(true_model, d, k_true, k_candidates, n_train, n_test, seed):
    """
    Запуск эксперимента для одного seed.
    Использует фиксированную true_model, генерирует только новые данные.
    
    Parameters
    ----------
    true_model : KOrderDependencyGaussian
        Фиксированная истинная модель
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    k_candidates : list
        Список значений k для тестирования
    n_train : int
        Размер обучающей выборки
    n_test : int
        Размер тестовой выборки
    seed : int
        Случайное состояние для генерации данных
    
    Returns
    -------
    dict : словарь с результатами для всех k
    """
    # Создаем генератор с конкретным seed для данных
    rng = np.random.RandomState(seed)
    
    # Генерируем данные (используем rng для генерации)
    # Заменяем rng в true_model временно
    old_rng = true_model.rng
    true_model.rng = rng
    X_train = true_model.sample(n_train)
    X_test = true_model.sample(n_test)
    true_model.rng = old_rng  # Восстанавливаем
    
    log_p_true = true_model.log_prob(X_test)
    true_ll = log_p_true.mean()

    seed_results = {"true_ll": true_ll}

    for k in k_candidates:
        my_model = MyModel(d=d, k=k)
        my_model.fit(X_train)
        log_p_my = my_model.log_prob(X_test)

        mean_loglik = log_p_my.mean()

        # Подсчет параметров
        total_params = my_model.count_parameters()

        # AIC и BIC
        aic = -2 * mean_loglik + 2 * total_params
        bic = -2 * mean_loglik + np.log(n_train) * total_params

        # Log-likelihood per parameter
        loglik_per_param = mean_loglik / total_params if total_params > 0 else 0

        # Устойчивость параметров
        avg_sigma_sq = my_model.get_avg_sigma_squared()

        kl_est = (log_p_true - log_p_my).mean()

        # JS divergence
        p_unnorm = np.exp(log_p_true - log_p_true.max())
        q_unnorm = np.exp(log_p_my - log_p_my.max())
        p_hat = p_unnorm / p_unnorm.sum()
        q_hat = q_unnorm / q_unnorm.sum()
        m_hat = 0.5 * (p_hat + q_hat)
        eps = 1e-12
        p_safe = np.clip(p_hat, eps, 1.0)
        q_safe = np.clip(q_hat, eps, 1.0)
        m_safe = np.clip(m_hat, eps, 1.0)
        kl_p_m = np.sum(p_safe * (np.log(p_safe) - np.log(m_safe)))
        kl_q_m = np.sum(q_safe * (np.log(q_safe) - np.log(m_safe)))
        js_est = 0.5 * (kl_p_m + kl_q_m)

        seed_results[k] = {
            "mean_loglik": mean_loglik,
            "kl": kl_est,
            "js": js_est,
            "aic": aic,
            "bic": bic,
            "loglik_per_param": loglik_per_param,
            "avg_sigma_sq": avg_sigma_sq,
            "total_params": total_params,
        }

    return seed_results


def _run_single_seed_d_experiment(seed, d, k, n_train, n_test):
    """
    Запуск эксперимента для одного seed с фиксированным d и k.
    
    Parameters
    ----------
    seed : int
        Случайное состояние
    d : int
        Размерность данных
    k : int
        Порядок зависимости
    n_train : int
        Размер обучающей выборки
    n_test : int
        Размер тестовой выборки
    
    Returns
    -------
    dict : словарь с результатами
    """
    true_model = KOrderDependencyGaussian(k=k, d=d, random_state=seed)
    X_train = true_model.sample(n_train)
    X_test = true_model.sample(n_test)
    log_p_true = true_model.log_prob(X_test)
    true_ll = log_p_true.mean()

    my_model = MyModel(d=d, k=k)
    my_model.fit(X_train)
    log_p_my = my_model.log_prob(X_test)

    mean_loglik = log_p_my.mean()

    # Подсчет параметров
    total_params = my_model.count_parameters()

    # AIC и BIC
    aic = -2 * mean_loglik + 2 * total_params
    bic = -2 * mean_loglik + np.log(n_train) * total_params

    # Log-likelihood per parameter
    loglik_per_param = mean_loglik / total_params if total_params > 0 else 0

    # Устойчивость параметров
    avg_sigma_sq = my_model.get_avg_sigma_squared()

    kl_est = (log_p_true - log_p_my).mean()

    # JS divergence
    p_unnorm = np.exp(log_p_true - log_p_true.max())
    q_unnorm = np.exp(log_p_my - log_p_my.max())
    p_hat = p_unnorm / p_unnorm.sum()
    q_hat = q_unnorm / q_unnorm.sum()
    m_hat = 0.5 * (p_hat + q_hat)
    eps = 1e-12
    p_safe = np.clip(p_hat, eps, 1.0)
    q_safe = np.clip(q_hat, eps, 1.0)
    m_safe = np.clip(m_hat, eps, 1.0)
    kl_p_m = np.sum(p_safe * (np.log(p_safe) - np.log(m_safe)))
    kl_q_m = np.sum(q_safe * (np.log(q_safe) - np.log(m_safe)))
    js_est = 0.5 * (kl_p_m + kl_q_m)

    return {
        "true_ll": true_ll,
        "mean_loglik": mean_loglik,
        "kl": kl_est,
        "js": js_est,
        "aic": aic,
        "bic": bic,
        "loglik_per_param": loglik_per_param,
        "avg_sigma_sq": avg_sigma_sq,
        "total_params": total_params,
    }


def run_k_experiment_parallel(
    d=8, k_true=3, n_train=5000, n_test=5000, n_repeats=30, k_candidates=None, n_jobs=-1
):
    """
    Параллельный эксперимент по перебору k для MyModel с progress bar.
    
    Истинная модель создается один раз, данные генерируются с разными seed.
    
    Parameters
    ----------
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости в истинной модели
    n_train : int
        Размер обучающей выборки
    n_test : int
        Размер тестовой выборки
    n_repeats : int
        Число повторов (разные сиды для данных)
    k_candidates : list, optional
        Список k для MyModel (по умолчанию range(d))
    n_jobs : int
        Число параллельных процессов (-1 для всех доступных)
    
    Returns
    -------
    dict : результаты эксперимента
    """
    if k_candidates is None:
        k_candidates = list(range(d))

    seeds = list(range(n_repeats))

    # ВАЖНО: Создаём ОДНУ true_model для ВСЕХ seed
    true_model = KOrderDependencyGaussian(k=k_true, d=d, random_state=42)

    # Создаём список задач
    tasks = [
        delayed(_run_single_seed)(
            true_model,
            d,
            k_true,
            k_candidates,
            n_train,
            n_test,
            seed,
        )
        for seed in seeds
    ]

    print(f"Running {n_repeats} seeds with d={d}, k_true={k_true}...")
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

    # Агрегация результатов
    ks = np.array(k_candidates)
    mean_loglik_mat = np.zeros((n_repeats, len(ks)))
    kl_mat = np.zeros_like(mean_loglik_mat)
    js_mat = np.zeros_like(mean_loglik_mat)
    aic_mat = np.zeros_like(mean_loglik_mat)
    bic_mat = np.zeros_like(mean_loglik_mat)
    loglik_per_param_mat = np.zeros_like(mean_loglik_mat)
    avg_sigma_sq_mat = np.zeros_like(mean_loglik_mat)
    total_params_mat = np.zeros_like(mean_loglik_mat)
    true_lls = []

    for i, res in enumerate(all_results):
        true_lls.append(res["true_ll"])
        for j, k in enumerate(ks):
            mean_loglik_mat[i, j] = res[k]["mean_loglik"]
            kl_mat[i, j] = res[k]["kl"]
            js_mat[i, j] = res[k]["js"]
            aic_mat[i, j] = res[k]["aic"]
            bic_mat[i, j] = res[k]["bic"]
            loglik_per_param_mat[i, j] = res[k]["loglik_per_param"]
            avg_sigma_sq_mat[i, j] = res[k]["avg_sigma_sq"]
            total_params_mat[i, j] = res[k]["total_params"]

    # Доверительные интервалы (mean ± 1.96*std для 95% CI)
    mean_loglik_mean = mean_loglik_mat.mean(axis=0)
    mean_loglik_std = mean_loglik_mat.std(axis=0)
    mean_loglik_lo = mean_loglik_mean - 1.96 * mean_loglik_std
    mean_loglik_hi = mean_loglik_mean + 1.96 * mean_loglik_std

    kl_mean = kl_mat.mean(axis=0)
    kl_std = kl_mat.std(axis=0)
    kl_lo = kl_mean - 1.96 * kl_std
    kl_hi = kl_mean + 1.96 * kl_std

    js_mean = js_mat.mean(axis=0)
    js_std = js_mat.std(axis=0)
    js_lo = js_mean - 1.96 * js_std
    js_hi = js_mean + 1.96 * js_std

    aic_mean = aic_mat.mean(axis=0)
    aic_std = aic_mat.std(axis=0)
    aic_lo = aic_mean - 1.96 * aic_std
    aic_hi = aic_mean + 1.96 * aic_std

    bic_mean = bic_mat.mean(axis=0)
    bic_std = bic_mat.std(axis=0)
    bic_lo = bic_mean - 1.96 * bic_std
    bic_hi = bic_mean + 1.96 * bic_std

    loglik_per_param_mean = loglik_per_param_mat.mean(axis=0)
    loglik_per_param_std = loglik_per_param_mat.std(axis=0)
    loglik_per_param_lo = loglik_per_param_mean - 1.96 * loglik_per_param_std
    loglik_per_param_hi = loglik_per_param_mean + 1.96 * loglik_per_param_std

    avg_sigma_sq_mean = avg_sigma_sq_mat.mean(axis=0)
    avg_sigma_sq_std = avg_sigma_sq_mat.std(axis=0)
    avg_sigma_sq_lo = avg_sigma_sq_mean - 1.96 * avg_sigma_sq_std
    avg_sigma_sq_hi = avg_sigma_sq_mean + 1.96 * avg_sigma_sq_std

    total_params_mean = total_params_mat.mean(axis=0)

    true_ll_mean = np.mean(true_lls)

    # Визуализация - 2x4 сетка
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 1. Log-likelihood
    axes[0, 0].plot(ks, mean_loglik_mean, label="Mean log-likelihood", color="C0")
    axes[0, 0].scatter(ks, mean_loglik_mean, color="C0", s=40, zorder=3)
    axes[0, 0].fill_between(
        ks, mean_loglik_lo, mean_loglik_hi, color="C0", alpha=0.2, label="95% CI"
    )
    axes[0, 0].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[0, 0].axhline(
        y=true_ll_mean, color="red", linestyle="-", label="True log-likelihood"
    )
    axes[0, 0].set_xlabel("k (MyModel parameter)")
    axes[0, 0].set_ylabel("Mean test log-likelihood")
    axes[0, 0].set_title(f"Log-likelihood vs k (d={d}, k_true={k_true})")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. AIC
    axes[0, 1].plot(ks, aic_mean, label="AIC", color="C1")
    axes[0, 1].scatter(ks, aic_mean, color="C1", s=40, zorder=3)
    axes[0, 1].fill_between(ks, aic_lo, aic_hi, color="C1", alpha=0.2, label="95% CI")
    axes[0, 1].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("AIC (lower is better)")
    axes[0, 1].set_title(f"AIC vs k (d={d}, k_true={k_true})")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. BIC
    axes[0, 2].plot(ks, bic_mean, label="BIC", color="C2")
    axes[0, 2].scatter(ks, bic_mean, color="C2", s=40, zorder=3)
    axes[0, 2].fill_between(ks, bic_lo, bic_hi, color="C2", alpha=0.2, label="95% CI")
    axes[0, 2].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[0, 2].set_xlabel("k")
    axes[0, 2].set_ylabel("BIC (lower is better)")
    axes[0, 2].set_title(f"BIC vs k (d={d}, k_true={k_true})")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Количество параметров
    axes[0, 3].plot(
        ks, total_params_mean, label="Total parameters", color="C5", marker="o"
    )
    axes[0, 3].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[0, 3].set_xlabel("k")
    axes[0, 3].set_ylabel("Number of parameters")
    axes[0, 3].set_title(f"Model Complexity vs k (d={d})")
    axes[0, 3].legend()
    axes[0, 3].grid(True)

    # 5. KL divergence
    axes[1, 0].plot(ks, kl_mean, label="KL(P_true || P_k)", color="C6")
    axes[1, 0].scatter(ks, kl_mean, color="C6", s=40, zorder=3)
    axes[1, 0].fill_between(ks, kl_lo, kl_hi, color="C6", alpha=0.2, label="95% CI")
    axes[1, 0].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Estimated KL divergence")
    axes[1, 0].set_title(f"KL(P_true || P_k) vs k (d={d}, k_true={k_true})")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 6. JS divergence
    axes[1, 1].plot(ks, js_mean, label="JS(P_true || P_k)", color="C7")
    axes[1, 1].scatter(ks, js_mean, color="C7", s=40, zorder=3)
    axes[1, 1].fill_between(ks, js_lo, js_hi, color="C7", alpha=0.2, label="95% CI")
    axes[1, 1].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Estimated JS divergence")
    axes[1, 1].set_title(f"JS divergence vs k (d={d}, k_true={k_true})")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 7. Log-likelihood per parameter
    axes[1, 2].plot(ks, loglik_per_param_mean, label="LL per parameter", color="C3")
    axes[1, 2].scatter(ks, loglik_per_param_mean, color="C3", s=40, zorder=3)
    axes[1, 2].fill_between(
        ks,
        loglik_per_param_lo,
        loglik_per_param_hi,
        color="C3",
        alpha=0.2,
        label="95% CI",
    )
    axes[1, 2].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[1, 2].set_xlabel("k")
    axes[1, 2].set_ylabel("Log-likelihood per parameter")
    axes[1, 2].set_title(f"LL per parameter vs k (d={d}, k_true={k_true})")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # 8. Устойчивость параметров (avg sigma^2)
    axes[1, 3].plot(ks, avg_sigma_sq_mean, label="Avg σ²", color="C4")
    axes[1, 3].scatter(ks, avg_sigma_sq_mean, color="C4", s=40, zorder=3)
    axes[1, 3].fill_between(
        ks, avg_sigma_sq_lo, avg_sigma_sq_hi, color="C4", alpha=0.2, label="95% CI"
    )
    axes[1, 3].axvline(x=k_true, color="k", linestyle="--", label="true k")
    axes[1, 3].set_xlabel("k")
    axes[1, 3].set_ylabel("Average σ² (parameter stability)")
    axes[1, 3].set_title(f"Parameter Stability vs k (d={d}, k_true={k_true})")
    axes[1, 3].legend()
    axes[1, 3].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "ks": ks,
        "mean_loglik_mat": mean_loglik_mat,
        "kl_mat": kl_mat,
        "js_mat": js_mat,
        "aic_mat": aic_mat,
        "bic_mat": bic_mat,
        "loglik_per_param_mat": loglik_per_param_mat,
        "avg_sigma_sq_mat": avg_sigma_sq_mat,
        "total_params_mat": total_params_mat,
        "true_ll_mean": true_ll_mean,
    }


def run_d_experiment_parallel(
    d_candidates, k_fixed, n_train=5000, n_test=5000, n_repeats=30, n_jobs=-1
):
    """
    Параллельный эксперимент по перебору d для фиксированного k.

    Parameters
    ----------
    d_candidates : list
        Список значений d для тестирования
    k_fixed : int
        Фиксированное значение k (порядок зависимости)
    n_train : int
        Размер обучающей выборки
    n_test : int
        Размер тестовой выборки
    n_repeats : int
        Число повторов (разные сиды)
    n_jobs : int
        Число параллельных процессов (-1 для всех доступных)

    Returns
    -------
    dict : результаты эксперимента
    """
    seeds = list(range(n_repeats))

    # Создаём список задач
    tasks = []
    for d in d_candidates:
        # k не может быть больше d-1
        k_actual = min(k_fixed, d - 1)
        for seed in seeds:
            tasks.append(
                delayed(_run_single_seed_d_experiment)(
                    seed, d, k_actual, n_train, n_test
                )
            )

    print(f"Running {len(tasks)} tasks (d in {d_candidates}, k_fixed={k_fixed})...")
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

    # Агрегация результатов
    ds = np.array(d_candidates)
    n_d = len(ds)

    mean_loglik_mat = np.zeros((n_repeats, n_d))
    kl_mat = np.zeros_like(mean_loglik_mat)
    js_mat = np.zeros_like(mean_loglik_mat)
    true_ll_mat = np.zeros_like(mean_loglik_mat)
    aic_mat = np.zeros_like(mean_loglik_mat)
    bic_mat = np.zeros_like(mean_loglik_mat)
    loglik_per_param_mat = np.zeros_like(mean_loglik_mat)
    avg_sigma_sq_mat = np.zeros_like(mean_loglik_mat)
    total_params_mat = np.zeros_like(mean_loglik_mat)

    idx = 0
    for j, d in enumerate(d_candidates):
        for i in range(n_repeats):
            res = all_results[idx]
            mean_loglik_mat[i, j] = res["mean_loglik"]
            kl_mat[i, j] = res["kl"]
            js_mat[i, j] = res["js"]
            true_ll_mat[i, j] = res["true_ll"]
            aic_mat[i, j] = res["aic"]
            bic_mat[i, j] = res["bic"]
            loglik_per_param_mat[i, j] = res["loglik_per_param"]
            avg_sigma_sq_mat[i, j] = res["avg_sigma_sq"]
            total_params_mat[i, j] = res["total_params"]
            idx += 1

    # Доверительные интервалы (mean ± 1.96*std для 95% CI)
    mean_loglik_mean = mean_loglik_mat.mean(axis=0)
    mean_loglik_std = mean_loglik_mat.std(axis=0)
    mean_loglik_lo = mean_loglik_mean - 1.96 * mean_loglik_std
    mean_loglik_hi = mean_loglik_mean + 1.96 * mean_loglik_std

    kl_mean = kl_mat.mean(axis=0)
    kl_std = kl_mat.std(axis=0)
    kl_lo = kl_mean - 1.96 * kl_std
    kl_hi = kl_mean + 1.96 * kl_std

    js_mean = js_mat.mean(axis=0)
    js_std = js_mat.std(axis=0)
    js_lo = js_mean - 1.96 * js_std
    js_hi = js_mean + 1.96 * js_std

    true_ll_mean = true_ll_mat.mean(axis=0)
    true_ll_std = true_ll_mat.std(axis=0)
    true_ll_lo = true_ll_mean - 1.96 * true_ll_std
    true_ll_hi = true_ll_mean + 1.96 * true_ll_std

    aic_mean = aic_mat.mean(axis=0)
    aic_std = aic_mat.std(axis=0)
    aic_lo = aic_mean - 1.96 * aic_std
    aic_hi = aic_mean + 1.96 * aic_std

    bic_mean = bic_mat.mean(axis=0)
    bic_std = bic_mat.std(axis=0)
    bic_lo = bic_mean - 1.96 * bic_std
    bic_hi = bic_mean + 1.96 * bic_std

    loglik_per_param_mean = loglik_per_param_mat.mean(axis=0)
    loglik_per_param_std = loglik_per_param_mat.std(axis=0)
    loglik_per_param_lo = loglik_per_param_mean - 1.96 * loglik_per_param_std
    loglik_per_param_hi = loglik_per_param_mean + 1.96 * loglik_per_param_std

    avg_sigma_sq_mean = avg_sigma_sq_mat.mean(axis=0)
    avg_sigma_sq_std = avg_sigma_sq_mat.std(axis=0)
    avg_sigma_sq_lo = avg_sigma_sq_mean - 1.96 * avg_sigma_sq_std
    avg_sigma_sq_hi = avg_sigma_sq_mean + 1.96 * avg_sigma_sq_std

    total_params_mean = total_params_mat.mean(axis=0)

    # Визуализация - 2x4 сетка
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 1. Log-likelihood
    axes[0, 0].plot(
        ds,
        mean_loglik_mean,
        "o-",
        label="Estimated LL",
        color="C0",
        linewidth=2,
        markersize=8,
    )
    axes[0, 0].fill_between(
        ds,
        mean_loglik_lo,
        mean_loglik_hi,
        color="C0",
        alpha=0.2,
        label="95% CI (estimated)",
    )
    axes[0, 0].plot(
        ds, true_ll_mean, "s--", label="True LL", color="red", linewidth=2, markersize=8
    )
    axes[0, 0].fill_between(
        ds, true_ll_lo, true_ll_hi, color="red", alpha=0.1, label="95% CI (true)"
    )
    axes[0, 0].set_xlabel("Dimension d")
    axes[0, 0].set_ylabel("Mean test log-likelihood")
    axes[0, 0].set_title(f"Log-likelihood vs d (k={k_fixed})")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. AIC
    axes[0, 1].plot(
        ds, aic_mean, "o-", label="AIC", color="C1", linewidth=2, markersize=8
    )
    axes[0, 1].fill_between(ds, aic_lo, aic_hi, color="C1", alpha=0.2, label="95% CI")
    axes[0, 1].set_xlabel("Dimension d")
    axes[0, 1].set_ylabel("AIC (lower is better)")
    axes[0, 1].set_title(f"AIC vs d (k={k_fixed})")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. BIC
    axes[0, 2].plot(
        ds, bic_mean, "o-", label="BIC", color="C2", linewidth=2, markersize=8
    )
    axes[0, 2].fill_between(ds, bic_lo, bic_hi, color="C2", alpha=0.2, label="95% CI")
    axes[0, 2].set_xlabel("Dimension d")
    axes[0, 2].set_ylabel("BIC (lower is better)")
    axes[0, 2].set_title(f"BIC vs d (k={k_fixed})")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Количество параметров
    axes[0, 3].plot(
        ds,
        total_params_mean,
        "o-",
        label="Total parameters",
        color="C5",
        linewidth=2,
        markersize=8,
    )
    axes[0, 3].set_xlabel("Dimension d")
    axes[0, 3].set_ylabel("Number of parameters")
    axes[0, 3].set_title(f"Model Complexity vs d (k={k_fixed})")
    axes[0, 3].legend()
    axes[0, 3].grid(True)

    # 5. KL divergence
    axes[1, 0].plot(
        ds,
        kl_mean,
        "o-",
        label="KL(P_true || P_est)",
        color="C6",
        linewidth=2,
        markersize=8,
    )
    axes[1, 0].fill_between(ds, kl_lo, kl_hi, color="C6", alpha=0.2, label="95% CI")
    axes[1, 0].set_xlabel("Dimension d")
    axes[1, 0].set_ylabel("Estimated KL divergence")
    axes[1, 0].set_title(f"KL divergence vs d (k={k_fixed})")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 6. JS divergence
    axes[1, 1].plot(
        ds,
        js_mean,
        "o-",
        label="JS(P_true || P_est)",
        color="C7",
        linewidth=2,
        markersize=8,
    )
    axes[1, 1].fill_between(ds, js_lo, js_hi, color="C7", alpha=0.2, label="95% CI")
    axes[1, 1].set_xlabel("Dimension d")
    axes[1, 1].set_ylabel("Estimated JS divergence")
    axes[1, 1].set_title(f"JS divergence vs d (k={k_fixed})")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 7. Log-likelihood per parameter
    axes[1, 2].plot(
        ds,
        loglik_per_param_mean,
        "o-",
        label="LL per parameter",
        color="C3",
        linewidth=2,
        markersize=8,
    )
    axes[1, 2].fill_between(
        ds,
        loglik_per_param_lo,
        loglik_per_param_hi,
        color="C3",
        alpha=0.2,
        label="95% CI",
    )
    axes[1, 2].set_xlabel("Dimension d")
    axes[1, 2].set_ylabel("Log-likelihood per parameter")
    axes[1, 2].set_title(f"LL per parameter vs d (k={k_fixed})")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # 8. Устойчивость параметров
    axes[1, 3].plot(
        ds,
        avg_sigma_sq_mean,
        "o-",
        label="Avg σ²",
        color="C4",
        linewidth=2,
        markersize=8,
    )
    axes[1, 3].fill_between(
        ds, avg_sigma_sq_lo, avg_sigma_sq_hi, color="C4", alpha=0.2, label="95% CI"
    )
    axes[1, 3].set_xlabel("Dimension d")
    axes[1, 3].set_ylabel("Average σ² (parameter stability)")
    axes[1, 3].set_title(f"Parameter Stability vs d (k={k_fixed})")
    axes[1, 3].legend()
    axes[1, 3].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "ds": ds,
        "k_fixed": k_fixed,
        "mean_loglik_mat": mean_loglik_mat,
        "kl_mat": kl_mat,
        "js_mat": js_mat,
        "true_ll_mat": true_ll_mat,
        "aic_mat": aic_mat,
        "bic_mat": bic_mat,
        "loglik_per_param_mat": loglik_per_param_mat,
        "avg_sigma_sq_mat": avg_sigma_sq_mat,
        "total_params_mat": total_params_mat,
    }


# ============================================================================
# Эксперименты с вариацией train size (оригинальные функции)
# ============================================================================

import random
import time
from functools import partial
from multiprocessing import Pool, cpu_count


def _run_single_repeat_train_size(
    repeat,
    true_model_class,
    true_model_kwargs,
    models_to_compare,
    d,
    k_true,
    train_sizes,
    n_test,
):
    """
    Запустить один повтор эксперимента для разных размеров train
    """
    np.random.seed(repeat)
    random.seed(repeat)

    true_model = true_model_class(
        k=k_true, d=d, random_state=repeat, **true_model_kwargs
    )

    # Генерировать максимальный объем данных
    max_train = max(train_sizes)
    X_train_full = true_model.sample(max_train)
    X_test = true_model.sample(n_test)
    log_p_true = true_model.log_prob(X_test)

    repeat_results = {}

    for n_train in train_sizes:
        X_train = X_train_full[:n_train]
        repeat_results[n_train] = {}

        for model_name, ModelClass in models_to_compare.items():
            try:
                # Инициализация модели
                if model_name == "ARModel_Average":
                    model = ModelClass(
                        d=d, k=k_true, n_random_orders=10, random_state=repeat
                    )
                elif "_k=" in model_name:
                    # Generic: parse k from name like "MyModel_k=3" or "ParencliticModel_k=3"
                    k_for_model = int(model_name.split("=")[-1])
                    model = ModelClass(d=d, k=k_for_model)
                else:
                    model = ModelClass(d=d, k=k_true)

                model.fit(X_train)
                log_p_est = model.log_prob(X_test)

                mean_loglik = log_p_est.mean()
                kl = (log_p_true - log_p_est).mean()

                # JS divergence
                eps = 1e-12
                p_unnorm = np.exp(np.clip(log_p_true - log_p_true.max(), -700, 0))
                q_unnorm = np.exp(np.clip(log_p_est - log_p_est.max(), -700, 0))
                p_hat = p_unnorm / p_unnorm.sum()
                q_hat = q_unnorm / q_unnorm.sum()
                m_hat = 0.5 * (p_hat + q_hat)
                p_safe = np.clip(p_hat, eps, 1.0)
                q_safe = np.clip(q_hat, eps, 1.0)
                m_safe = np.clip(m_hat, eps, 1.0)
                kl_p_m = np.sum(p_safe * (np.log(p_safe) - np.log(m_safe)))
                kl_q_m = np.sum(q_safe * (np.log(q_safe) - np.log(m_safe)))
                js = 0.5 * (kl_p_m + kl_q_m)

                total_params = model.count_parameters()
                aic = -2 * mean_loglik + 2 * total_params
                bic = -2 * mean_loglik + np.log(n_train) * total_params

                repeat_results[n_train][model_name] = {
                    "loglik": mean_loglik,
                    "kl": kl,
                    "js": js,
                    "aic": aic,
                    "bic": bic,
                }
            except Exception as e:
                repeat_results[n_train][model_name] = {
                    "loglik": np.nan,
                    "kl": np.nan,
                    "js": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                }

    return repeat, repeat_results


def run_train_size_experiment(
    true_model_class,
    true_model_kwargs,
    models_to_compare,
    d,
    k_true,
    train_sizes,
    n_test,
    n_repeats,
    n_jobs=None,
):
    """
    Эксперимент: сравнение моделей при разных размерах train data
    
    Parameters
    ----------
    true_model_class : class
        Класс истинной модели (например, KOrderDependencyGaussian)
    true_model_kwargs : dict
        Параметры для истинной модели
    models_to_compare : dict
        Словарь {model_name: ModelClass} моделей для сравнения
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    train_sizes : list
        Список размеров обучающей выборки
    n_test : int
        Размер тестовой выборки
    n_repeats : int
        Число повторов
    n_jobs : int, optional
        Число параллельных процессов (по умолчанию cpu_count())
    
    Returns
    -------
    results : dict
        Результаты {metric: {model_name: {n_train: [values]}}}
    train_sizes : list
        Использованные размеры обучающей выборки
    model_names : list
        Имена моделей
    """
    if n_jobs is None or n_jobs == -1:
        n_jobs = cpu_count()

    print(
        f"Running train size experiment: {len(train_sizes)} sizes × {n_repeats} repeats"
    )
    print(f"Train sizes: {train_sizes}")
    print(f"Using {n_jobs} parallel jobs")

    run_partial = partial(
        _run_single_repeat_train_size,
        true_model_class=true_model_class,
        true_model_kwargs=true_model_kwargs,
        models_to_compare=models_to_compare,
        d=d,
        k_true=k_true,
        train_sizes=train_sizes,
        n_test=n_test,
    )

    # Запуск с отслеживанием прогресса
    all_results = []
    start_time = time.time()

    with Pool(processes=n_jobs) as pool:
        results_iter = pool.imap(run_partial, range(n_repeats))

        for i, result in enumerate(results_iter):
            all_results.append(result)

            # Вычисление прогресса и оставшегося времени
            elapsed = time.time() - start_time
            completed = i + 1
            avg_time_per_task = elapsed / completed
            remaining_tasks = n_repeats - completed
            eta = avg_time_per_task * remaining_tasks

            # Форматирование времени
            eta_min, eta_sec = divmod(int(eta), 60)
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)

            print(
                f"\rProgress: {completed}/{n_repeats} ({100*completed/n_repeats:.1f}%) | "
                f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d} | "
                f"ETA: {eta_min:02d}:{eta_sec:02d}",
                end="",
                flush=True,
            )

    print()  # Новая строка после завершения
    total_time = time.time() - start_time
    print(f"✓ Completed in {total_time:.1f}s")

    # Агрегация результатов
    metrics = ["loglik", "kl", "js", "aic", "bic"]
    model_names = list(models_to_compare.keys())

    results = {
        metric: {
            model_name: {n_train: [] for n_train in train_sizes}
            for model_name in model_names
        }
        for metric in metrics
    }

    for repeat, repeat_results in all_results:
        for n_train in train_sizes:
            for model_name in model_names:
                for metric in metrics:
                    val = repeat_results[n_train][model_name][metric]
                    results[metric][model_name][n_train].append(val)

    return results, train_sizes, model_names


# ============================================================================
# Эксперименты с вариацией уровня шума (оригинальные функции)
# ============================================================================

def _run_single_repeat_noise_level(
    repeat,
    true_model_class,
    d,
    k_true,
    noise_levels,
    models_to_compare,
    n_train,
    n_test,
):
    """
    Запустить один повтор эксперимента для разных уровней шума.

    Шум добавляется к X_train (имитация шумных наблюдений при обучении).
    X_test и log_p_true — ЧИСТЫЕ, общие для всех уровней шума. Так корректно
    измеряется KL(p_true || p_model), где p_model обучалась на шумных данных.
    """
    np.random.seed(repeat)
    random.seed(repeat)

    # ВАЖНО: истинная модель и чистый тест создаются ОДИН раз
    true_model = true_model_class(
        k=k_true,
        d=d,
        mu_min=-1,
        mu_max=1,
        sigma=1,
        random_state=repeat,
    )

    X_train_clean = true_model.sample(n_train)
    X_test = true_model.sample(n_test)  # чистый тест
    log_p_true = true_model.log_prob(X_test)  # считаем один раз на чистом тесте
    true_ll_mean = log_p_true.mean()

    repeat_results = {}

    for noise_level in noise_levels:
        # Добавляем шум ТОЛЬКО к train (тест остаётся чистым)
        X_train = X_train_clean + noise_level * np.random.randn(*X_train_clean.shape)

        repeat_results[noise_level] = {}

        for model_name, ModelClass in models_to_compare.items():
            try:
                # Инициализация модели
                if model_name == "ARModel_Average":
                    model = ModelClass(
                        d=d, k=k_true, n_random_orders=10, random_state=repeat
                    )
                elif "_k=" in model_name:
                    k_for_model = int(model_name.split("=")[-1])
                    model = ModelClass(d=d, k=k_for_model)
                else:
                    model = ModelClass(d=d, k=k_true)

                model.fit(X_train)
                log_p_est = model.log_prob(X_test)

                mean_loglik = log_p_est.mean()
                kl = (log_p_true - log_p_est).mean()

                # JS divergence
                eps = 1e-12
                p_unnorm = np.exp(np.clip(log_p_true - log_p_true.max(), -700, 0))
                q_unnorm = np.exp(np.clip(log_p_est - log_p_est.max(), -700, 0))
                p_hat = p_unnorm / p_unnorm.sum()
                q_hat = q_unnorm / q_unnorm.sum()
                m_hat = 0.5 * (p_hat + q_hat)
                p_safe = np.clip(p_hat, eps, 1.0)
                q_safe = np.clip(q_hat, eps, 1.0)
                m_safe = np.clip(m_hat, eps, 1.0)
                kl_p_m = np.sum(p_safe * (np.log(p_safe) - np.log(m_safe)))
                kl_q_m = np.sum(q_safe * (np.log(q_safe) - np.log(m_safe)))
                js = 0.5 * (kl_p_m + kl_q_m)

                total_params = model.count_parameters()
                aic = -2 * mean_loglik + 2 * total_params
                bic = -2 * mean_loglik + np.log(n_train) * total_params

                rel_error = np.abs(mean_loglik - true_ll_mean) / np.abs(true_ll_mean)

                repeat_results[noise_level][model_name] = {
                    "loglik": mean_loglik,
                    "true_loglik": true_ll_mean,
                    "kl": kl,
                    "js": js,
                    "aic": aic,
                    "bic": bic,
                    "rel_error": rel_error,
                }
            except Exception as e:
                repeat_results[noise_level][model_name] = {
                    "loglik": np.nan,
                    "true_loglik": np.nan,
                    "kl": np.nan,
                    "js": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "rel_error": np.nan,
                }

    return repeat, repeat_results


def run_noise_level_experiment(
    true_model_class,
    models_to_compare,
    d,
    k_true,
    noise_levels,
    n_train,
    n_test,
    n_repeats,
    n_jobs=None,
):
    """
    Эксперимент: сравнение моделей при разных уровнях шума в данных.
    
    Parameters
    ----------
    true_model_class : class
        Класс истинной модели (например, KOrderDependencyGaussian)
    models_to_compare : dict
        Словарь {model_name: ModelClass} моделей для сравнения
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    noise_levels : list
        Список уровней шума (std добавляемого шума)
    n_train : int
        Размер обучающей выборки
    n_test : int
        Размер тестовой выборки
    n_repeats : int
        Число повторов
    n_jobs : int, optional
        Число параллельных процессов (по умолчанию cpu_count())
    
    Returns
    -------
    results : dict
        Результаты {metric: {model_name: {noise_level: [values]}}}
    noise_levels : list
        Использованные уровни шума
    model_names : list
        Имена моделей
    """
    if n_jobs is None or n_jobs == -1:
        n_jobs = cpu_count()

    print(f"Running noise level experiment: {len(noise_levels)} levels × {n_repeats} repeats")
    print(f"Noise levels (std_min): {noise_levels}")
    print(f"Using {n_jobs} parallel jobs")

    run_partial = partial(
        _run_single_repeat_noise_level,
        true_model_class=true_model_class,
        d=d,
        k_true=k_true,
        noise_levels=noise_levels,
        models_to_compare=models_to_compare,
        n_train=n_train,
        n_test=n_test,
    )

    # Запуск с отслеживанием прогресса
    all_results = []
    start_time = time.time()

    with Pool(processes=n_jobs) as pool:
        results_iter = pool.imap(run_partial, range(n_repeats))

        for i, result in enumerate(results_iter):
            all_results.append(result)

            elapsed = time.time() - start_time
            completed = i + 1
            avg_time_per_task = elapsed / completed
            remaining_tasks = n_repeats - completed
            eta = avg_time_per_task * remaining_tasks

            eta_min, eta_sec = divmod(int(eta), 60)
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)

            print(
                f"\rProgress: {completed}/{n_repeats} ({100*completed/n_repeats:.1f}%) | "
                f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d} | "
                f"ETA: {eta_min:02d}:{eta_sec:02d}",
                end="",
                flush=True,
            )

    print()
    total_time = time.time() - start_time
    print(f"✓ Completed in {total_time:.1f}s")

    # Агрегация результатов
    metrics = ["loglik", "true_loglik", "kl", "js", "aic", "bic", "rel_error"]
    model_names = list(models_to_compare.keys())

    results = {
        metric: {
            model_name: {noise: [] for noise in noise_levels}
            for model_name in model_names
        }
        for metric in metrics
    }

    for repeat, repeat_results in all_results:
        for noise_level in noise_levels:
            for model_name in model_names:
                for metric in metrics:
                    val = repeat_results[noise_level][model_name][metric]
                    results[metric][model_name][noise_level].append(val)

    return results, noise_levels, model_names