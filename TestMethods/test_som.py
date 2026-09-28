"""
Тестовый скрипт для проверки SOM (PyClustering) с визуализацией
Автор: Курбанов Рамазан [peressmit@mail.ru]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.datasets import make_blobs, make_moons, make_circles

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ALGORITHM_NAME = "SOM"

try:
    from ClusteringMethods.SOMClustering import (
        ConcreteStrategySOM_from_PYCLUSTERING
    )
    from ClusteringMethods.ClasteringAlgorithms import (
        Context,
        StrategiesManager
    )
    ALGORITHM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Предупреждение: {e}")
    ALGORITHM_AVAILABLE = False

STRATEGY_AVAILABLE = ALGORITHM_AVAILABLE

TEST_DIR = Path(__file__).parent
IMAGES_DIR = TEST_DIR / "Images" / ALGORITHM_NAME
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"📁 Директория для результатов: {IMAGES_DIR}")
print("="*80 + "\n")


def generate_test_data_2d(dataset_type='blobs', n_samples=300, **kwargs):
    np.random.seed(kwargs.get('random_state', 42))
    if dataset_type == 'blobs':
        X, y_true = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=kwargs.get('centers', 3),
            cluster_std=kwargs.get('cluster_std', 0.5),
            random_state=kwargs.get('random_state', 42)
        )
    elif dataset_type == 'moons':
        X, y_true = make_moons(
            n_samples=n_samples,
            noise=kwargs.get('noise', 0.05),
            random_state=kwargs.get('random_state', 42)
        )
    elif dataset_type == 'circles':
        X, y_true = make_circles(
            n_samples=n_samples,
            noise=kwargs.get('noise', 0.05),
            factor=kwargs.get('factor', 0.5),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return X, y_true


def generate_test_data_3d(n_samples=300, n_clusters=3, cluster_std=0.5):
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=3,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )
    return X, y_true


def save_figure(fig, filename, dpi=150):
    output_path = IMAGES_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Сохранено: {output_path.name}")


def test_basic_2d_clustering():
    print("="*80)
    print("ТЕСТ 1: Базовая 2D Кластеризация")
    print("="*80)
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    X, y_true = generate_test_data_2d('blobs', n_samples=300, centers=3, cluster_std=0.5)
    config = StrategiesManager.getStrategyRunConfigById("som_pyc")
    strategy = ConcreteStrategySOM_from_PYCLUSTERING()
    y_pred = strategy.clastering_points(X, config)
    n_clusters_pred = len(np.unique(y_pred))
    n_clusters_true = len(np.unique(y_true))
    print(f"Количество точек: {len(X)}")
    print(f"Истинное количество кластеров: {n_clusters_true}")
    print(f"Найдено кластеров: {n_clusters_pred}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[0].set_title('Истинные Кластеры', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X₁', fontsize=12)
    axes[0].set_ylabel('X₂', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Истинная метка')
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[1].set_title(f'{ALGORITHM_NAME} Результаты\n({n_clusters_pred} кластеров)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X₁', fontsize=12)
    axes[1].set_ylabel('X₂', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Предсказанная метка')
    plt.tight_layout()
    save_figure(fig, 'test_basic_2d.png')
    print()


def test_parameter_sensitivity():
    print("="*80)
    print("ТЕСТ 2: Чувствительность к Параметрам (размер сетки rows x cols)")
    print("="*80)
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    X, y_true = generate_test_data_2d('blobs', n_samples=300, centers=3)
    grid_values = [(1, 2), (1, 3), (2, 2), (3, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    for i, (rows, cols) in enumerate(grid_values):
        config = StrategiesManager.getStrategyRunConfigById("som_pyc")
        config["rows"] = rows
        config["cols"] = cols
        strategy = ConcreteStrategySOM_from_PYCLUSTERING()
        y_pred = strategy.clastering_points(X, config)
        n_clusters = len(np.unique(y_pred))
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'сетка {rows}x{cols}\n({n_clusters} кластеров)', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].grid(True, alpha=0.3)
        print(f"  сетка {rows}x{cols}: {n_clusters} кластеров")
    plt.tight_layout()
    save_figure(fig, 'test_parameters.png')
    print()


def test_different_datasets():
    print("="*80)
    print("ТЕСТ 3: Разные Типы Датасетов")
    print("="*80)
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    datasets = [
        ("Blobs (Сферические)", 'blobs', {'centers': 3, 'cluster_std': 0.5}, 3),
        ("Moons (Полумесяцы)", 'moons', {'noise': 0.05}, 2),
        ("Circles (Круги)", 'circles', {'noise': 0.05, 'factor': 0.5}, 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, dataset_type, params, n_clusters) in enumerate(datasets):
        X, _ = generate_test_data_2d(dataset_type, n_samples=300, **params)
        config = StrategiesManager.getStrategyRunConfigById("som_pyc")
        config["cols"] = n_clusters
        strategy = ConcreteStrategySOM_from_PYCLUSTERING()
        y_pred = strategy.clastering_points(X, config)
        n_found = len(np.unique(y_pred))
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'{name}\n({n_found} кластеров)', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].grid(True, alpha=0.3)
        print(f"  {name}: {n_found} кластеров")
    plt.tight_layout()
    save_figure(fig, 'test_datasets.png')
    print()


def test_3d_clustering():
    print("="*80)
    print("ТЕСТ 4: 3D Кластеризация")
    print("="*80)
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    X, y_true = generate_test_data_3d(n_samples=300, n_clusters=3)
    config = StrategiesManager.getStrategyRunConfigById("som_pyc")
    strategy = ConcreteStrategySOM_from_PYCLUSTERING()
    y_pred = strategy.clastering_points(X, config)
    n_clusters = len(np.unique(y_pred))
    print(f"Найдено кластеров: {n_clusters}")
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap='viridis', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax1.set_title('3D: Истинные Кластеры', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('X₃')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax2.set_title(f'3D: {ALGORITHM_NAME}\n({n_clusters} кластеров)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('X₃')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    ax3 = fig.add_subplot(1, 3, 3)
    scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax3.set_title('2D Проекция (XY)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X₁')
    ax3.set_ylabel('X₂')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3)
    plt.tight_layout()
    save_figure(fig, 'test_3d.png')
    print()


def test_strategy_integration():
    print("="*80)
    print("ТЕСТ 5: Интеграция с Фреймворком (Strategy Pattern)")
    print("="*80)
    if not STRATEGY_AVAILABLE:
        print("❌ Strategy Pattern не доступен. Пропускаем тест.")
        return
    X, y_true = generate_test_data_2d('blobs', n_samples=200, centers=2)
    try:
        config = StrategiesManager.getStrategyRunConfigById("som_pyc")
        config["cols"] = 2
        strategy = ConcreteStrategySOM_from_PYCLUSTERING()
        context = Context(strategy)
        y_pred = context.do_some_clustering_points(X.T, config)
        n_clusters = len(np.unique(y_pred))
        print(f"Найдено кластеров: {n_clusters}")
        print("✅ Интеграция с фреймворком работает!")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        ax.set_title(f'{ALGORITHM_NAME} через Strategy Pattern\n({n_clusters} кластеров)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X₁', fontsize=12)
        ax.set_ylabel('X₂', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Кластер')
        plt.tight_layout()
        save_figure(fig, 'test_strategy.png')
    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_neuron_map():
    print("="*80)
    print("ТЕСТ 6: Карта Нейронов после Обучения")
    print("="*80)
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    from ClusteringMethods.SOMClustering import _train_network, _assign_to_neurons
    X, _ = generate_test_data_2d('blobs', n_samples=300, centers=3, cluster_std=0.8)
    rows, cols = 3, 3
    config = StrategiesManager.getStrategyRunConfigById("som_pyc")
    config["rows"] = rows
    config["cols"] = cols
    config["normalize"] = False
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, epochs in zip(axes, [1, 100]):
        config["epochs"] = epochs
        weights = _train_network(X, config)
        labels = _assign_to_neurons(X, weights)
        grid = weights.reshape(rows, cols, -1)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.4, s=25, edgecolors='k', linewidth=0.3)
        for r in range(rows):
            ax.plot(grid[r, :, 0], grid[r, :, 1], 'k-', lw=1.5)
        for c in range(cols):
            ax.plot(grid[:, c, 0], grid[:, c, 1], 'k-', lw=1.5)
        ax.scatter(weights[:, 0], weights[:, 1], c='red', s=160, edgecolors='k', zorder=3, label='Нейроны')
        n_clusters = len(np.unique(labels))
        ax.set_title(f'Сетка {rows}x{cols}, эпох: {epochs}\n({n_clusters} кластеров)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.grid(True, alpha=0.3)
        ax.legend()
        print(f"  эпох {epochs}: {n_clusters} кластеров")
    plt.tight_layout()
    save_figure(fig, 'test_neuron_map.png')
    print()


def main():
    print("\n" + "="*80)
    print(f"ТЕСТИРОВАНИЕ АЛГОРИТМА: {ALGORITHM_NAME}")
    print("="*80 + "\n")
    if not ALGORITHM_AVAILABLE:
        print("❌ ОШИБКА: Алгоритм не найден!")
        return
    try:
        test_basic_2d_clustering()
        test_parameter_sensitivity()
        test_different_datasets()
        test_3d_clustering()
        test_strategy_integration()
        test_neuron_map()
        print("="*80)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО! ✅")
        print("="*80)
        print(f"\n📊 Результаты сохранены в: {IMAGES_DIR}")
        print(f"📁 Всего файлов: {len(list(IMAGES_DIR.glob('*.png')))}")
        print("\nОткройте изображения для просмотра результатов!")
        print("="*80 + "\n")
        plt.show()
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
