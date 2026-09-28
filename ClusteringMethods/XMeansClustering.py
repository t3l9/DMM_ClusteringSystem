"""
X-Means Algorithm Implementation
Автор: Курбанов Рамазан [peressmit@mail.ru]
Последнее обновление: 2026-09-28

Кластеризация методом X-Means на базе pyclustering.cluster.xmeans.

Алгоритм — расширение KMeans, которое само подбирает число кластеров.
Работа начинается с небольшого числа центров (kmin). Затем повторяется:
1. Обычный KMeans с текущими центрами.
2. Каждый кластер пробуем разделить на два (локальный KMeans с k=2).
3. Разделение сохраняется, только если оно улучшает критерий качества
   (BIC или MNDL), то есть если два кластера описывают данные заметно
   лучше одного с учётом "штрафа" за лишние параметры.
Алгоритм останавливается, когда ни одно разделение не улучшает
критерий или число кластеров достигло kmax.

Ограничения:
- pyclustering обращается к numpy.warnings, удалённому в numpy >= 2.
  Для совместимости numpy.warnings подменяется стандартным модулем
  warnings (только если атрибута нет).
- Критерии BIC и MNDL рассчитаны на круглые (сферические) кластеры
  одинакового разброса. Вытянутые кластеры алгоритм дробит на части,
  поэтому нормализация по умолчанию выключена: она сжимает оси
  по-разному и может сделать круглые кластеры вытянутыми.
- C++ часть pyclustering (ccore) собрана не для всех платформ
  (например, отсутствует для macOS arm64). Если её не удаётся
  загрузить, используется реализация на Python.

Источник:
https://pyclustering.github.io/docs/0.10.1/html/dd/db4/classpyclustering_1_1cluster_1_1xmeans_1_1xmeans.html
"""

import warnings

import numpy as np
from pyclustering.cluster.xmeans import xmeans, splitting_type
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


# pyclustering (center_initializer) вызывает numpy.warnings.filterwarnings,
# а в numpy >= 2 этот псевдоним стандартного модуля warnings удалён.
if not hasattr(np, "warnings"):
    np.warnings = warnings


_CRITERIA = {
    "bic": splitting_type.BAYESIAN_INFORMATION_CRITERION,
    "mndl": splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH,
}


def _prepare_features(data: np.ndarray) -> np.ndarray:
    """
    Приводит входные данные к массиву (n_samples, n_features).

    Parameters:
    -----------
    data : array-like
        Точки или пиксели. Допускается одномерный вектор
        или транспонированный вид (n_features, n_samples),
        если признаков не больше 10.

    Returns:
    --------
    ndarray, shape (n_samples, n_features)
        Массив признаков типа float64.
    """
    features = np.asarray(data, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[0] < features.shape[1] and features.shape[0] <= 10:
        features = features.T
    return features


def _maybe_normalize(features: np.ndarray, normalize: bool) -> np.ndarray:
    """
    Стандартизирует признаки, если включена нормализация.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Исходные признаки.
    normalize : bool
        Если True, применяется StandardScaler.

    Returns:
    --------
    ndarray, shape (n_samples, n_features)
        Нормализованные или исходные признаки.
    """
    if not normalize:
        return features
    return StandardScaler().fit_transform(features)


def _labels_from_clusters(clusters, n_samples: int) -> np.ndarray:
    """
    Преобразует список кластеров pyclustering в массив меток.

    Parameters:
    -----------
    clusters : list of list of int
        Индексы точек каждого кластера.
    n_samples : int
        Общее число точек.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки.
    """
    labels = np.zeros(n_samples, dtype=np.intp)
    for cluster_index, cluster in enumerate(clusters):
        labels[cluster] = cluster_index
    return labels


def _fit_predict(features: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
    """
    Запускает X-Means и возвращает метки.

    Сначала пробует C++ реализацию (если включена), при ошибке её
    загрузки переключается на Python-реализацию.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки.
    params : StrategyRunConfig
        Параметры стратегии.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки.
    """
    n_samples = len(features)
    if n_samples == 0:
        return np.array([], dtype=np.intp)
    if n_samples == 1:
        return np.array([0], dtype=np.intp)

    kmin = max(1, min(int(params["kmin"]), n_samples))
    kmax = max(kmin, min(int(params["kmax"]), n_samples))
    random_state = int(params["random_state"])

    initial, _ = kmeans_plusplus(features, kmin, random_state=random_state)

    data = features.tolist()
    kwargs = {
        "kmax": kmax,
        "tolerance": float(params["tolerance"]),
        "criterion": _CRITERIA.get(str(params["criterion"]), splitting_type.BAYESIAN_INFORMATION_CRITERION),
        "repeat": max(1, int(params["repeat"])),
        "random_state": random_state,
    }

    try:
        model = xmeans(data, initial.tolist(), ccore=bool(params["ccore"]), **kwargs)
        model.process()
    except OSError:
        model = xmeans(data, initial.tolist(), ccore=False, **kwargs)
        model.process()

    return _labels_from_clusters(model.get_clusters(), n_samples)


@StrategiesManager.registerStrategy(
    "xmeans_pyc",
    "X-Means (PyClustering)",
    "KMeans с автоматическим подбором числа кластеров из pyclustering"
)
class ConcreteStrategyXMeans_from_PYCLUSTERING(Strategy):
    """
    Стратегия кластеризации X-Means из pyclustering.

    Parameters:
    -----------
    kmin : int, default=2
        Начальное количество кластеров.

    kmax : int, default=20
        Максимальное количество кластеров.

    criterion : {'bic', 'mndl'}, default='bic'
        Критерий, по которому решается, делить ли кластер на два.

    tolerance : float, default=0.001
        Точность остановки внутреннего KMeans.

    repeat : int, default=1
        Сколько раз запускать KMeans при попытке разделения кластера.

    random_state : int, default=42
        Инициализация генератора случайных чисел.

    normalize : bool, default=False
        Стандартизировать признаки перед кластеризацией. По умолчанию
        выключено: стандартизация сжимает оси по-разному и делает круглые
        кластеры вытянутыми, из-за чего X-Means дробит их на части.

    ccore : bool, default=True
        Использовать C++ часть pyclustering (с откатом на Python).
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров, отображаемых в GUI."""
        cls._addParam(
            "kmin",
            "Начальное количество кластеров",
            StrategyParamType.UNumber,
            """
            С какого числа кластеров начинается поиск.
            Дальше алгоритм сам делит кластеры, пока это улучшает критерий.
            Рекомендуется: 1-2.
            """,
            2
        )

        cls._addParam(
            "kmax",
            "Максимальное количество кластеров",
            StrategyParamType.UNumber,
            """
            Верхняя граница числа кластеров. Алгоритм не создаст больше.
            """,
            20
        )

        cls._addParam(
            "criterion",
            "Критерий разделения",
            StrategyParamType.Switch,
            """
            Критерий, по которому решается, делить ли кластер на два:

            - bic: байесовский информационный критерий (рекомендуется)
            - mndl: минимальная длина описания без шума
              (обычно находит меньше кластеров)
            """,
            "bic",
            switches=["bic", "mndl"]
        )

        cls._addParam(
            "tolerance",
            "Точность остановки",
            StrategyParamType.UFloating,
            """
            Внутренний KMeans останавливается, когда максимальный сдвиг
            центров за итерацию меньше этого значения.
            """,
            0.001
        )

        cls._addParam(
            "repeat",
            "Число повторов KMeans",
            StrategyParamType.UNumber,
            """
            Сколько раз запускать KMeans при попытке разделить кластер.
            Больше — надёжнее, но медленнее.
            """,
            1
        )

        cls._addParam(
            "random_state",
            "Состояние случайности",
            StrategyParamType.UNumber,
            """
            Инициализация генератора случайных чисел для воспроизводимости результатов.
            """,
            42
        )

        cls._addParam(
            "normalize",
            "Нормализовать данные",
            StrategyParamType.Bool,
            """
            Стандартизировать признаки перед кластеризацией.
            Включайте только для признаков разного масштаба: стандартизация
            сжимает оси по-разному, круглые кластеры становятся вытянутыми,
            и X-Means (критерий рассчитан на круглые кластеры) дробит их.
            """,
            False
        )

        cls._addParam(
            "ccore",
            "Использовать C++",
            StrategyParamType.Bool,
            """
            Использовать C++ часть библиотеки pyclustering для ускорения.
            Если она недоступна на текущей платформе, автоматически
            используется реализация на Python.
            """,
            True
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения методом X-Means.

        Parameters:
        -----------
        pixels : ndarray
            Пиксели изображения в виде массива признаков.
        params : StrategyRunConfig
            Параметры запуска стратегии.

        Returns:
        --------
        ndarray
            Метка кластера для каждого пикселя.
        """
        pixels = _prepare_features(pixels)
        pixels_proc = _maybe_normalize(pixels, bool(params["normalize"]))
        return _fit_predict(pixels_proc, params)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом X-Means.

        Parameters:
        -----------
        points : ndarray
            Точки данных в виде массива (n_samples, n_features).
        params : StrategyRunConfig
            Параметры запуска стратегии.

        Returns:
        --------
        ndarray
            Метка кластера для каждой точки.
        """
        points = _prepare_features(points)
        points_proc = _maybe_normalize(points, bool(params["normalize"]))
        return _fit_predict(points_proc, params)
