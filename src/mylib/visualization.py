"""
Функции визуализации для экспериментов с k-order зависимостями.

Функции:
- plot_model_comparison_boxplots: Boxplots для сравнения моделей
- plot_model_comparison_violins: Violin plots для сравнения моделей
- plot_pairwise_comparison: Попарное сравнение MyModel vs baselines
- convert_results_to_dataframe: Конвертация результатов в DataFrame
- create_summary_table: Создание сводной таблицы
- create_model_comparison_table: Создание таблицы сравнения по метрике
- run_full_analysis: Полный анализ результатов
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def convert_results_to_dataframe(results, n_repeats):
    """
    Конвертировать результаты из dict формата в pandas DataFrame.

    Parameters
    ----------
    results : dict
        Структура: results[model_name][metric_name] = list of values (n_repeats)
    n_repeats : int
        Количество повторов

    Returns
    -------
    df : pd.DataFrame
        Таблица с колонками: model, metric, repeat, value
    """
    data_list = []

    for model_name, metrics_dict in results.items():
        for metric_name, values_list in metrics_dict.items():
            for repeat_idx, value in enumerate(values_list):
                data_list.append(
                    {
                        "model": model_name,
                        "metric": metric_name,
                        "repeat": repeat_idx,
                        "value": value,
                    }
                )

    df = pd.DataFrame(data_list)
    return df


def create_summary_table(df):
    """
    Создать сводную таблицу со статистикой по моделям и метрикам.

    Parameters
    ----------
    df : pd.DataFrame
        Результаты всех экспериментов

    Returns
    -------
    summary : pd.DataFrame
        Таблица: model, metric, mean, std, q025, q975, min, max
    """
    summary = (
        df.groupby(["model", "metric"])["value"]
        .agg(
            [
                ("count", "count"),
                ("mean", "mean"),
                ("median", "median"),
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
                ("q025", lambda x: np.percentile(x, 2.5)),
                ("q975", lambda x: np.percentile(x, 97.5)),
                ("q250", lambda x: np.percentile(x, 25.0)),
                ("q750", lambda x: np.percentile(x, 75.0)),
            ]
        )
        .reset_index()
    )

    return summary


def create_model_comparison_table(summary, metric_name, best_is_min=True):
    """
    Создать таблицу сравнения моделей для конкретной метрики.

    Parameters
    ----------
    summary : pd.DataFrame
        Сводная таблица
    metric_name : str
        Имя метрики (например, 'kl', 'aic', 'loglik')
    best_is_min : bool
        True если лучше минимум, False если максимум

    Returns
    -------
    table : pd.DataFrame
        Отформатированная таблица
    """
    metric_data = summary[summary["metric"] == metric_name][
        ["model", "mean", "std", "q025", "q975"]
    ].copy()

    # Отсортировать по 'mean' (лучшие наверху)
    metric_data = metric_data.sort_values("mean", ascending=best_is_min).reset_index(
        drop=True
    )

    # Добавить рейтинг
    metric_data.insert(0, "rank", range(1, len(metric_data) + 1))

    # Форматирование: mean ± std [q025, q975]
    metric_data["mean±std"] = metric_data.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )
    metric_data["95% CI"] = metric_data.apply(
        lambda row: f"[{row['q025']:.4f}, {row['q975']:.4f}]", axis=1
    )

    # Оставить только нужные колонки
    display_table = metric_data[["rank", "model", "mean±std", "95% CI"]]
    display_table.columns = ["Rank", "Model", "Mean ± Std", "95% Confidence Interval"]

    return display_table


def plot_model_comparison_boxplots(df, d, k_true, n_repeats, save=True):
    """
    Создать 3x2 фигуру с boxplots для 6 основных метрик.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с колонками: model, metric, repeat, value
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    n_repeats : int
        Количество повторов
    save : bool
        Сохранять ли фигуру в файл
    """

    # Метрики для визуализации и их параметры
    metrics_config = [
        {"key": "loglik", "title": "Test Log-likelihood (выше лучше)", "pos": (0, 0)},
        {"key": "val_loglik", "title": "Validation Log-likelihood (выше лучше)", "pos": (0, 1)},
        {"key": "kl", "title": "KL(P_true || P_est) (ниже лучше)", "pos": (1, 0)},
        {"key": "js", "title": "JS divergence (ниже лучше)", "pos": (1, 1)},
        {"key": "aic", "title": "AIC (ниже лучше)", "pos": (2, 0)},
        {"key": "loglik_per_param", "title": "Efficiency: loglik/param (выше лучше)", "pos": (2, 1)},
    ]

    # Создать фигуру
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))

    # Цветовая палитра для моделей
    model_colors = {
        "MyModel": "#1f77b4",  # синий
        "ARModel_Fixed": "#ff7f0e",  # оранжевый
        "ARModel_Random": "#d62728",  # красный
        "ARModel_Average": "#9467bd",  # фиолетовый
        "Gaussian_Full": "#2ca02c",  # зелёный
    }

    for config in metrics_config:
        metric_key = config["key"]
        row, col = config["pos"]
        ax = axs[row, col]

        # Фильтровать данные для этой метрики
        metric_df = df[df["metric"] == metric_key].copy()

        if len(metric_df) == 0:
            ax.text(
                0.5, 0.5, f"No data for {metric_key}",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(config["title"])
            continue

        # Boxplot
        models_list = sorted(metric_df["model"].unique())

        # Подготовить данные для boxplot
        data_to_plot = [
            metric_df[metric_df["model"] == model]["value"].values
            for model in models_list
        ]

        bp = ax.boxplot(
            data_to_plot,
            labels=models_list,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="red",
                markeredgecolor="darkred",
                markersize=8,
                label="Mean",
            ),
        )

        # Раскрасить боксы
        for patch, model in zip(bp["boxes"], models_list):
            # Подобрать цвет для модели
            color = "gray"
            for key in model_colors:
                if key in model:
                    color = model_colors[key]
                    break
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Выделить MyModel жирной границей
        for patch, model in zip(bp["boxes"], models_list):
            if "MyModel" in model:
                patch.set_linewidth(3)
                patch.set_edgecolor("darkblue")

        # Оформление
        ax.set_title(config["title"], fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    # Общий заголовок
    fig.suptitle(
        f"Сравнение моделей плотности (d={d}, k_true={k_true}, n_repeats={n_repeats})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    
    if save:
        filename = f"boxplot_comparison_d{d}_ktrue{k_true}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Сохранена фигура: {filename}")
    
    plt.show()


def _convert_comparison_results_to_df(comparison_results):
    """
    Конвертировать результаты run_model_comparison_experiment_parallel в DataFrame.
    
    Parameters
    ----------
    comparison_results : dict
        Результаты от run_model_comparison_experiment_parallel
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame с колонками: model, metric, repeat, value
    """
    model_names = comparison_results['model_names']
    n_repeats = len(comparison_results[model_names[0]]['mean_loglik'])
    
    data_list = []
    
    for model_name in model_names:
        model_data = comparison_results[model_name]
        for metric in ['mean_loglik', 'aic', 'bic', 'kl', 'total_params']:
            if metric in model_data:
                for i, value in enumerate(model_data[metric]):
                    # Переименуем mean_loglik -> loglik для совместимости
                    metric_name = 'loglik' if metric == 'mean_loglik' else metric
                    data_list.append({
                        'model': model_name,
                        'metric': metric_name,
                        'repeat': i,
                        'value': value
                    })
    
    return pd.DataFrame(data_list)


def plot_model_comparison_violins(results_or_df, d=None, k_true=None, n_repeats=None, save=True):
    """
    Создать 3x2 фигуру с violin plots для лучшей видимости распределения.

    Parameters
    ----------
    results_or_df : dict or pd.DataFrame
        Либо результаты от run_model_comparison_experiment_parallel,
        либо DataFrame с колонками: model, metric, repeat, value
    d : int, optional
        Размерность данных
    k_true : int, optional
        Истинный порядок зависимости
    n_repeats : int, optional
        Количество повторов
    save : bool
        Сохранять ли фигуру в файл
    """
    # Конвертируем в DataFrame если нужно
    if isinstance(results_or_df, dict):
        df = _convert_comparison_results_to_df(results_or_df)
        params = results_or_df.get('params', {})
        if d is None:
            d = params.get('d', '?')
        if k_true is None:
            k_true = params.get('k_true', '?')
        if n_repeats is None:
            n_repeats = params.get('n_repeats', '?')
    else:
        df = results_or_df
        if d is None:
            d = '?'
        if k_true is None:
            k_true = '?'
        if n_repeats is None:
            n_repeats = len(df['repeat'].unique()) if 'repeat' in df.columns else '?'
    
    if not HAS_SEABORN:
        print("Warning: seaborn not available, using boxplots instead")
        plot_model_comparison_boxplots(df, d, k_true, n_repeats, save)
        return

    metrics_config = [
        {"key": "loglik", "title": "Test Log-likelihood", "pos": (0, 0)},
        {"key": "val_loglik", "title": "Validation Log-likelihood", "pos": (0, 1)},
        {"key": "kl", "title": "KL divergence", "pos": (1, 0)},
        {"key": "js", "title": "JS divergence", "pos": (1, 1)},
        {"key": "aic", "title": "AIC", "pos": (2, 0)},
        {"key": "loglik_per_param", "title": "Loglik/Param", "pos": (2, 1)},
    ]

    fig, axs = plt.subplots(3, 2, figsize=(16, 14))

    model_colors = {
        "MyModel": "#1f77b4",
        "ARModel_Fixed": "#ff7f0e",
        "ARModel_Random": "#d62728",
        "ARModel_Average": "#9467bd",
        "Gaussian_Full": "#2ca02c",
    }

    # Расширяем цвета для моделей MyModel с разными k
    for mod_name in df["model"].unique():
        if "MyModel" in mod_name and mod_name not in model_colors:
            model_colors[mod_name] = "#1f77b4"

    for config in metrics_config:
        metric_key = config["key"]
        row, col = config["pos"]
        ax = axs[row, col]

        metric_df = df[df["metric"] == metric_key].copy()

        if len(metric_df) == 0:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Violin plot через seaborn
        sns.violinplot(
            data=metric_df,
            x="model",
            y="value",
            palette=model_colors,
            ax=ax,
            inner="quartile",
            cut=0,
        )

        # Добавить точки для каждого повтора
        sns.stripplot(
            data=metric_df,
            x="model",
            y="value",
            color="black",
            alpha=0.3,
            size=3,
            ax=ax,
        )

        ax.set_title(config["title"], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Распределение метрик (d={d}, k_true={k_true})",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    
    if save:
        filename = f"violin_comparison_d{d}_ktrue{k_true}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Сохранена фигура: {filename}")
    
    plt.show()


def plot_pairwise_comparison(results_or_df, d=None, k_true=None, save=True):
    """
    Создать фигуру с попарным сравнением MyModel vs каждый бэйзлайн.

    Parameters
    ----------
    results_or_df : dict or pd.DataFrame
        Либо результаты от run_model_comparison_experiment_parallel,
        либо DataFrame с колонками: model, metric, repeat, value
    d : int, optional
        Размерность данных
    k_true : int, optional
        Истинный порядок зависимости
    save : bool
        Сохранять ли фигуру в файл
    """
    # Конвертируем в DataFrame если нужно
    if isinstance(results_or_df, dict):
        df = _convert_comparison_results_to_df(results_or_df)
        params = results_or_df.get('params', {})
        if d is None:
            d = params.get('d', '?')
        if k_true is None:
            k_true = params.get('k_true', '?')
    else:
        df = results_or_df
        if d is None:
            d = '?'
        if k_true is None:
            k_true = '?'

    metrics_config = [
        {"key": "loglik", "title": "Log-likelihood", "pos": (0, 0)},
        {"key": "kl", "title": "KL divergence", "pos": (0, 1)},
        {"key": "aic", "title": "AIC", "pos": (0, 2)},
        {"key": "bic", "title": "BIC", "pos": (1, 0)},
        {"key": "total_params", "title": "Parameters", "pos": (1, 1)},
    ]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    baseline_models = [
        "AR_Fixed",
        "AR_Random",
        "AR_Average",
        "Gaussian_Full",
    ]

    for config in metrics_config:
        metric_key = config["key"]
        row, col = config["pos"]
        ax = axs[row, col]

        metric_df = df[df["metric"] == metric_key].copy()

        if len(metric_df) == 0:
            ax.text(
                0.5, 0.5, f"No data for {metric_key}",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(config["title"], fontsize=12, fontweight="bold")
            continue
        # Получить данные MyModel
        my_unique_models = list(set(
            metric_df[metric_df["model"].apply(lambda x: "MyModel" in x)]["model"].values
        ))

        mymodel_data = [
            metric_df[metric_df["model"] == name]["value"].values
            for name in my_unique_models
        ]

        if len(mymodel_data) == 0:
            ax.text(
                0.5, 0.5, "No MyModel data",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(config["title"], fontsize=12, fontweight="bold")
            continue

        # Подготовить данные для боксплотов
        data_to_plot = list(mymodel_data)
        x_positions = list(range(len(my_unique_models)))
        x_labels = list(my_unique_models)
        colors_list = ["#1f77b4"] * len(my_unique_models)  # синий для MyModel

        # Добавить боксплоты для бэйзлайнов
        x_pos = len(x_positions)
        for baseline_model in baseline_models:
            baseline_data = metric_df[metric_df["model"] == baseline_model]["value"].values

            if len(baseline_data) > 0:
                data_to_plot.append(baseline_data)
                x_positions.append(x_pos)

                # Красивое форматирование названия модели
                label = baseline_model.replace("ARModel_", "").replace("_", "\n")
                x_labels.append(label)
                colors_list.append("#ff7f0e")  # оранжевый
                x_pos += 1

        # Создать боксплот со всеми данными разом
        bp = ax.boxplot(
            data_to_plot,
            positions=x_positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="red",
                markeredgecolor="darkred",
                markersize=7,
            ),
        )

        # Раскрасить боксы
        for idx, (patch, color) in enumerate(zip(bp["boxes"], colors_list)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

            # Выделить MyModel жирной границей
            if idx < len(my_unique_models):
                patch.set_linewidth(3)
                patch.set_edgecolor("darkblue")
            else:
                patch.set_linewidth(1.5)

        # Оформление
        ax.set_title(config["title"], fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    # Убрать пустой subplot
    axs[1, 2].axis("off")

    fig.suptitle(
        f"Сравнение: MyModel (синий) vs бэйзлайны (оранжевый)\n"
        f"d={d}, k_true={k_true}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    
    if save:
        filename = f"pairwise_comparison_d{d}_ktrue{k_true}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Сохранена фигура: {filename}")

    plt.show()


def run_full_analysis(results, d, k_true, n_repeats, output_prefix="experiment"):
    """
    Запустить полный анализ: таблицы + все графики.

    Parameters
    ----------
    results : dict
        Словарь результатов {model_name: {metric_name: [values]}}
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    n_repeats : int
        Количество повторов
    output_prefix : str
        Префикс для имён файлов

    Returns
    -------
    df : pd.DataFrame
        DataFrame с результатами
    summary : pd.DataFrame
        Сводная таблица
    """

    print("=" * 100)
    print("ЗАПУСК ПОЛНОГО АНАЛИЗА РЕЗУЛЬТАТОВ")
    print("=" * 100)

    # 1. Конвертирование в DataFrame
    print("\n[1/6] Конвертирование результатов в DataFrame...")
    df = convert_results_to_dataframe(results, n_repeats)
    summary = create_summary_table(df)

    # 2. Сохранить CSV
    print("[2/6] Сохранение результатов в CSV...")
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    df.to_csv(f"{output_prefix}_all_repeats.csv", index=False)
    print(f"  ✓ Сохранены: {output_prefix}_summary.csv, {output_prefix}_all_repeats.csv")

    # 3. Печать таблиц
    print("\n[3/6] Печать таблиц сравнения...")
    metrics_to_compare = {
        "loglik": ("Test Log-likelihood", False),
        "kl": ("KL divergence", True),
        "js": ("JS divergence", True),
        "aic": ("AIC", True),
        "loglik_per_param": ("Efficiency", False),
    }

    for metric_key, (metric_name, best_is_min) in metrics_to_compare.items():
        print(f"\n  {metric_name}:")
        table = create_model_comparison_table(summary, metric_key, best_is_min=best_is_min)
        print(table.to_string(index=False))

    # 4. Boxplot
    print("\n[4/6] Создание boxplot-графиков...")
    plot_model_comparison_boxplots(df, d, k_true, n_repeats)

    # 5. Violin plot
    print("[5/6] Создание violin-plot графиков...")
    plot_model_comparison_violins(df, d, k_true, n_repeats)

    # 6. Pairwise comparison
    print("[6/6] Создание попарного сравнения...")
    plot_pairwise_comparison(df, d, k_true)

    print("\n" + "=" * 100)
    print("✓ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 100)

    return df, summary


def plot_train_size_comparison(results, train_sizes, model_names, d, k_true, save=True):
    """
    Построить графики сравнения моделей при разных размерах train data.

    Parameters
    ----------
    results : dict
        Результаты эксперимента {metric: {model: {n_train: [values]}}}
    train_sizes : list
        Размеры обучающей выборки
    model_names : list
        Имена моделей
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    save : bool
        Сохранять ли фигуру
    """
    metrics_config = [
        ("loglik", "Test Log-likelihood", False),
        ("kl", "KL divergence", True),
        ("js", "JS divergence", True),
        ("aic", "AIC", True),
        ("bic", "BIC", True),
    ]

    # Используем качественную палитру с хорошо различимыми цветами
    n_models = len(model_names)
    if n_models <= 10:
        colors = plt.cm.tab10.colors[:n_models]
    elif n_models <= 20:
        colors = plt.cm.tab20.colors[:n_models]
    else:
        colors = plt.cm.hsv(np.linspace(0, 0.9, n_models))

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_name, lower_better) in enumerate(metrics_config):
        ax = axes[idx]

        y_s = []

        for i, model_name in enumerate(model_names):
            means = []
            lows = []
            highs = []

            for n_train in train_sizes:
                values = np.array(results[metric_key][model_name][n_train])
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    mean = np.mean(values)
                    std = np.std(values)
                    means.append(mean)
                    lows.append(mean - 1.96 * std)
                    highs.append(mean + 1.96 * std)
                else:
                    means.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)

            ax.plot(
                train_sizes,
                means,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=8,
                label=model_name,
                linewidth=2.5,
            )

            y_values = np.array(means)
            y_values_clean = y_values[~np.isnan(y_values)]
            y_values_clean = y_values_clean[~np.isinf(y_values_clean)]
            y_s += list(y_values_clean)

            ax.fill_between(train_sizes, lows, highs, color=colors[i], alpha=0.15)

        if len(y_s) > 0:
            y_min = np.percentile(y_s, 10)
            y_max = np.percentile(y_s, 90)
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_xlabel("Train data size", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    # Удалить лишний subplot
    axes[-1].axis("off")

    fig.suptitle(
        f"Model comparison vs Train data size (d={d}, k_true={k_true})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    
    if save:
        filename = f"train_size_comparison_d{d}_k{k_true}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {filename}")
    
    plt.show()


def plot_noise_level_comparison(results, noise_levels, model_names, d, k_true, n_train, save=True):
    """
    Построить графики сравнения моделей при разных уровнях шума.

    Parameters
    ----------
    results : dict
        Результаты эксперимента {metric: {model: {noise: [values]}}}
    noise_levels : list
        Уровни шума
    model_names : list
        Имена моделей
    d : int
        Размерность данных
    k_true : int
        Истинный порядок зависимости
    n_train : int
        Размер обучающей выборки
    save : bool
        Сохранять ли фигуру
    """
    metrics_config = [
        ("loglik", "Test Log-likelihood", False),
        ("kl", "KL divergence", True),
        ("js", "JS divergence", True),
        ("rel_error", "Relative Error |LL_est - LL_true| / |LL_true|", True),
        ("aic", "AIC", True),
        ("bic", "BIC", True),
    ]

    n_models = len(model_names)
    if n_models <= 10:
        colors = plt.cm.tab10.colors[:n_models]
    elif n_models <= 20:
        colors = plt.cm.tab20.colors[:n_models]
    else:
        colors = plt.cm.hsv(np.linspace(0, 0.9, n_models))

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_name, lower_better) in enumerate(metrics_config):
        ax = axes[idx]

        y_s = []

        for i, model_name in enumerate(model_names):
            means = []
            lows = []
            highs = []

            for noise in noise_levels:
                values = np.array(results[metric_key][model_name][noise])
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    mean = np.mean(values)
                    std = np.std(values)
                    means.append(mean)
                    lows.append(mean - 1.96 * std)
                    highs.append(mean + 1.96 * std)
                else:
                    means.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)

            ax.plot(
                noise_levels,
                means,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=8,
                label=model_name,
                linewidth=2.5,
            )

            y_values = np.array(means)
            y_values_clean = y_values[~np.isnan(y_values)]
            y_values_clean = y_values_clean[~np.isinf(y_values_clean)]
            y_s += list(y_values_clean)

            ax.fill_between(noise_levels, lows, highs, color=colors[i], alpha=0.15)

        if len(y_s) > 0:
            y_min = np.percentile(y_s, 10)
            y_max = np.percentile(y_s, 90)
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_xlabel("Noise level (std_min)", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"Model comparison vs Noise level (d={d}, k_true={k_true}, n_train={n_train})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    
    if save:
        filename = f"noise_level_comparison_d{d}_k{k_true}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {filename}")
    
    plt.show()
