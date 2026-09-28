"""
MBSAS Algorithm Implementation
Автор: Курбанов Рамазан [peressmit@mail.ru]
Последнее обновление: 2026-09-28

Кластеризация методом MBSAS (Modified Basic Sequential Algorithmic Scheme)
на базе pyclustering.cluster.mbsas.

Последовательный алгоритм, проходящий по данным дважды:
1. Первый проход — создание кластеров. Точки просматриваются по порядку.
   Если расстояние от точки до ближайшего представителя кластера больше
   порога threshold и лимит кластеров не исчерпан, точка открывает
   новый кластер. Остальные точки на этом проходе пропускаются.
2. Второй проход — распределение. Каждая пропущенная точка относится
   к ближайшему кластеру, а представитель кластера (среднее его точек)
   пересчитывается.
Число кластеров определяется порогом, но не превышает maximum_clusters.

Ограничения:
- Результат зависит от порядка точек: кластеры открывают точки,
  встретившиеся раньше. Параметр shuffle позволяет перемешать точки
  перед обработкой (с фиксированным random_state).
- Порог задаётся в единицах расстояния, поэтому зависит от масштаба
  данных. По умолчанию данные стандартизируются, и порог измеряется
  в стандартных отклонениях.
- C++ часть pyclustering (ccore) собрана не для всех платформ
  (например, отсутствует для macOS arm64). Если её не удаётся
  загрузить, используется реализация на Python.

Источник:
https://pyclustering.github.io/docs/0.10.1/html/d4/d02/classpyclustering_1_1cluster_1_1mbsas_1_1mbsas.html
"""

import numpy as np
from pyclustering.cluster.mbsas import mbsas
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


_METRICS = {
    "euclidean": type_metric.EUCLIDEAN,
    "manhattan": type_metric.MANHATTAN,
    "chebyshev": type_metric.CHEBYSHEV,
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
    Запускает MBSAS и возвращает метки.

    При shuffle=True точки обрабатываются в случайном (но
    воспроизводимом) порядке, метки возвращаются в исходном порядке.
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

    if bool(params["shuffle"]):
        order = np.random.default_rng(int(params["random_state"])).permutation(n_samples)
    else:
        order = np.arange(n_samples)

    data = features[order].tolist()
    maximum_clusters = max(1, int(params["maximum_clusters"]))
    threshold = float(params["threshold"])
    metric = distance_metric(_METRICS.get(str(params["metric"]), type_metric.EUCLIDEAN))

    try:
        model = mbsas(data, maximum_clusters, threshold, ccore=bool(params["ccore"]), metric=metric)
        model.process()
    except OSError:
        model = mbsas(data, maximum_clusters, threshold, ccore=False, metric=metric)
        model.process()

    labels = np.empty(n_samples, dtype=np.intp)
    labels[order] = _labels_from_clusters(model.get_clusters(), n_samples)
    return labels


@StrategiesManager.registerStrategy(
    "mbsas_pyc",
    "MBSAS (PyClustering)",
    "Модифицированный базовый последовательный алгоритм из pyclustering"
)
class ConcreteStrategyMBSAS_from_PYCLUSTERING(Strategy):
    """
    Стратегия кластеризации MBSAS из pyclustering.

    Parameters:
    -----------
    maximum_clusters : int, default=10
        Максимальное количество кластеров.

    threshold : float, default=1.0
        Порог расстояния: точка дальше порога от всех кластеров
        открывает новый кластер.

    metric : {'euclidean', 'manhattan', 'chebyshev'}, default='euclidean'
        Метрика расстояния.

    shuffle : bool, default=False
        Перемешать точки перед обработкой.

    random_state : int, default=42
        Инициализация генератора случайных чисел для перемешивания.

    normalize : bool, default=True
        Стандартизировать признаки перед кластеризацией.

    ccore : bool, default=True
        Использовать C++ часть pyclustering (с откатом на Python).
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров, отображаемых в GUI."""
        cls._addParam(
            "maximum_clusters",
            "Максимальное количество кластеров",
            StrategyParamType.UNumber,
            """
            Верхняя граница числа кластеров. Когда лимит исчерпан,
            новые кластеры не создаются, даже если точка дальше порога.
            """,
            10
        )

        cls._addParam(
            "threshold",
            "Порог расстояния",
            StrategyParamType.UFloating,
            """
            Если расстояние от точки до всех существующих кластеров больше
            порога, точка открывает новый кластер. Меньше порог — больше
            кластеров. При включённой нормализации измеряется
            в стандартных отклонениях. Рекомендуется: 0.5-1.5.
            """,
            1.0
        )

        cls._addParam(
            "metric",
            "Метрика расстояния",
            StrategyParamType.Switch,
            """
            Метрика для расчёта расстояния от точки до кластера.

            - euclidean: евклидово расстояние (рекомендуется)
            - manhattan: сумма модулей разностей
            - chebyshev: максимальная норма
            """,
            "euclidean",
            switches=["euclidean", "manhattan", "chebyshev"]
        )

        cls._addParam(
            "shuffle",
            "Перемешать точки",
            StrategyParamType.Bool,
            """
            Результат MBSAS зависит от порядка точек. Если включено,
            точки обрабатываются в случайном порядке (воспроизводимом
            при одинаковом random_state).
            """,
            False
        )

        cls._addParam(
            "random_state",
            "Состояние случайности",
            StrategyParamType.UNumber,
            """
            Инициализация генератора случайных чисел для перемешивания точек.
            """,
            42
        )

        cls._addParam(
            "normalize",
            "Нормализовать данные",
            StrategyParamType.Bool,
            """
            Стандартизировать признаки перед кластеризацией.
            Рекомендуется оставить включённым: тогда порог не зависит
            от масштаба данных.
            """,
            True
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
        Кластеризация изображения методом MBSAS.

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
        Кластеризация точек методом MBSAS.

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
