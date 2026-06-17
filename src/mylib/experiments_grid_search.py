"""
Эксперименты-переборы: поиск параметров генерации сфер, при которых
увеличение k значимо улучшает качество MyModelSynolitic.

Ключевое наблюдение:
  На «обычных» сферах GaussianNB уже при k=1 даёт ~99%, потому что
  маржинальное распределение каждого x_i различается между классами
  (разный радиус → разная дисперсия).

  Чтобы k>1 помогал, нужно **разрушить маржинальную информативность**,
  сохранив информацию в парах/тройках. Это достигается:

  1. **Случайным вращением** (random rotation) данных:
     после вращения каждый наблюдаемый признак — линейная комбинация
     исходных координат сферы, и маржинальные распределения классов
     перемешиваются. Пары координат сохраняют больше информации.

  2. **Перекрытием радиусов** (r_ctrl ∩ r_case):
     чем сильнее перекрытие, тем сложнее задача, и тем больше
     потенциал для выигрыша от k>1.

  3. **Малой размерностью сферы + высокой наблюдаемой размерностью**:
     сфера в 2–3D, вложенная в 8–15D через вращение + шум.
"""

import numpy as np
from itertools import product
from joblib import Parallel, delayed
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from models_synolitic import MyModelSynolitic
from classification import generate_sphere_dataset


# ─── Генерация данных с вращением ────────────────────────────────────────────

def generate_rotated_sphere(
    n_cases=400,
    n_controls=400,
    sphere_dims=3,
    observed_dims=10,
    noise_scale=0.0,
    r_ctrl=(0.01, 0.5),
    r_case=(0.5, 1.0),
    model="ideal",
    broken_fraction=0.5,
    random_state=42,
):
    """
    Генерирует сферические данные, а затем вращает и вкладывает в
    пространство большей размерности.

    1. Генерируем сферу в sphere_dims измерениях
    2. Дополняем нулями до observed_dims
    3. Применяем случайную ортогональную матрицу observed_dims × observed_dims
    4. (Опционально) добавляем гауссов шум

    Returns: X (n, observed_dims), y (n,)
    """
    rng = np.random.default_rng(random_state)

    # Генерируем сферу в малой размерности
    X_sphere, y, _ = generate_sphere_dataset(
        n_cases=n_cases,
        n_controls=n_controls,
        n_dims=sphere_dims,
        model=model,
        noise_dims=0,
        broken_fraction=broken_fraction,
        r_ctrl=r_ctrl,
        r_case=r_case,
        random_state=random_state,
    )

    n = X_sphere.shape[0]

    # Вкладываем в пространство большей размерности (дополняем нулями)
    X_embedded = np.zeros((n, observed_dims))
    X_embedded[:, :sphere_dims] = X_sphere

    # Случайное ортогональное вращение
    Q = ortho_group.rvs(observed_dims, random_state=rng)
    X_rotated = X_embedded @ Q.T

    # Добавляем шум
    if noise_scale > 0:
        X_rotated += rng.normal(0, noise_scale, size=X_rotated.shape)

    return X_rotated, y


# ─── Оценка одной конфигурации ───────────────────────────────────────────────

def evaluate_rotated_config(config: dict, k_values: list, n_repeats: int = 5) -> dict:
    """
    Оценивает качество для каждого k на данных с вращением.
    """
    results = {k: {"accs": [], "f1s": []} for k in k_values}

    for seed in range(n_repeats):
        X, y = generate_rotated_sphere(**config, random_state=42 + seed * 7)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42 + seed, stratify=y
        )
        d = config["observed_dims"]

        for k in k_values:
            m = MyModelSynolitic(
                d=d, k=k,
                classifier_class=QDA,
                clf_class_params={},
            )
            m.fit_parallel(X_tr, y_tr, n_jobs=1)
            y_pred = m.predict(X_te)
            results[k]["accs"].append(accuracy_score(y_te, y_pred))
            results[k]["f1s"].append(f1_score(y_te, y_pred, average="binary"))

    summary = {}
    for k in k_values:
        summary[k] = {
            "acc_mean": np.mean(results[k]["accs"]),
            "acc_std":  np.std(results[k]["accs"]),
            "f1_mean":  np.mean(results[k]["f1s"]),
            "f1_std":   np.std(results[k]["f1s"]),
        }

    return {"config": config, "results": summary}


# ─── Grid search ─────────────────────────────────────────────────────────────

def grid_search_rotated(
    k_values=(1, 2, 3),
    n_repeats=5,
    n_jobs=-1,
) -> list:
    """
    Перебор параметров вращённых сфер для поиска конфигураций,
    где k>1 значимо помогает.
    """
    configs = []
    for sphere_dims, observed_dims, noise_scale, r_overlap, model, broken_frac in product(
        [2, 3, 4, 5],                    # sphere_dims
        [6, 8, 10, 12],                  # observed_dims
        [0.0, 0.05, 0.1],               # noise_scale
        ["none", "partial", "strong"],   # r_overlap
        ["ideal", "broken", "hemisphere"],  # model
        [0.3, 0.5],                      # broken_fraction
    ):
        if sphere_dims >= observed_dims:
            continue
        if model != "broken" and broken_frac != 0.3:
            continue  # broken_fraction relevant only for broken

        radius_configs = {
            "none":    {"r_ctrl": (0.01, 0.4), "r_case": (0.6, 1.0)},
            "partial": {"r_ctrl": (0.01, 0.6), "r_case": (0.4, 1.0)},
            "strong":  {"r_ctrl": (0.1, 0.8),  "r_case": (0.3, 1.0)},
        }
        r_cfg = radius_configs[r_overlap]

        configs.append({
            "n_cases": 400,
            "n_controls": 400,
            "sphere_dims": sphere_dims,
            "observed_dims": observed_dims,
            "noise_scale": noise_scale,
            "model": model,
            "broken_fraction": broken_frac,
            **r_cfg,
        })

    print(f"Всего конфигураций: {len(configs)}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(evaluate_rotated_config)(cfg, list(k_values), n_repeats)
        for cfg in configs
    )
    return results


def find_best_k_improvement(results: list, k_base=1, k_target=2, metric="acc_mean"):
    """Находит конфигурации с наибольшим улучшением."""
    improvements = []
    for r in results:
        base_val = r["results"][k_base][metric]
        target_val = r["results"][k_target][metric]
        improvement = target_val - base_val
        improvements.append((improvement, r["config"], r["results"]))
    improvements.sort(key=lambda x: x[0], reverse=True)
    return improvements


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Запуск grid search (вращённые сферы)...")
    results = grid_search_rotated(k_values=(1, 2, 3), n_repeats=5, n_jobs=-1)

    for k_target in [2, 3]:
        print(f"\n{'='*80}")
        print(f"TOP-15 конфигураций по улучшению k=1 → k={k_target}:")
        print("=" * 80)
        top = find_best_k_improvement(results, k_base=1, k_target=k_target)
        for i, (imp, cfg, res) in enumerate(top[:15]):
            print(f"\n#{i+1}  Δacc = {imp:+.4f}")
            print(f"  model={cfg['model']}, sphere_d={cfg['sphere_dims']}, "
                  f"obs_d={cfg['observed_dims']}, noise={cfg['noise_scale']:.2f}, "
                  f"broken_frac={cfg['broken_fraction']}")
            print(f"  r_ctrl={cfg['r_ctrl']}, r_case={cfg['r_case']}")
            for k in [1, 2, 3]:
                print(f"    k={k}: acc={res[k]['acc_mean']:.4f}±{res[k]['acc_std']:.4f}  "
                      f"f1={res[k]['f1_mean']:.4f}±{res[k]['f1_std']:.4f}")
