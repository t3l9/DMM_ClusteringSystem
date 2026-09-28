"""
SOM (Self-Organizing Map) Algorithm Implementation
Автор: Курбанов Рамазан [peressmit@mail.ru]
Последнее обновление: 2026-09-28

Кластеризация самоорганизующейся картой Кохонена на базе
pyclustering.nnet.som.

SOM — нейронная сеть из нейронов, расположенных в узлах сетки
rows x cols. У каждого нейрона есть вектор весов — точка в пространстве
данных. Обучение повторяется epochs раз:
1. Для каждой точки находится нейрон-победитель — нейрон с ближайшими
   весами.
2. Веса победителя и его соседей по сетке сдвигаются в сторону точки.
   Сила сдвига (learning rate) и радиус соседства уменьшаются
   с каждой эпохой.
После обучения каждая точка относится к своему нейрону-победителю:
один нейрон — один кластер. Число кластеров не превышает rows * cols;
нейроны, не выигравшие ни одной точки, в результат не попадают.

Ограничения:
- Обучение на Python-реализации медленное, поэтому для изображений
  с числом пикселей больше _IMAGE_MAX_SAMPLES сеть обучается
  на равномерной выборке, а все пиксели относятся к ближайшему нейрону.
- C++ часть pyclustering (ccore) собрана не для всех платформ
  (например, отсутствует для macOS arm64). Если её не удаётся
  загрузить, используется реализация на Python.

Источник:
https://pyclustering.github.io/docs/0.10.1/html/d7/d2c/classpyclustering_1_1nnet_1_1som_1_1som.html
"""

import numpy as np
from pyclustering.nnet.som import som, som_parameters, type_conn, type_init
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


_IMAGE_MAX_SAMPLES = 3000

_CONNECTIONS = {
    "grid_eight": type_conn.grid_eight,
    "grid_four": type_conn.grid_four,
    "honeycomb": type_conn.honeycomb,
    "func_neighbor": type_conn.func_neighbor,
}

_INITIALIZATIONS = {
    "uniform_grid": type_init.uniform_grid,
    "random": type_init.random,
    "random_centroid": type_init.random_centroid,
    "random_surface": type_init.random_surface,
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


def _train_network(features: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
    """
    Обучает SOM и возвращает веса нейронов.

    Сначала пробует C++ реализацию (если включена), при ошибке её
    загрузки переключается на Python-реализацию.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Данные для обучения.
    params : StrategyRunConfig
        Параметры стратегии.

    Returns:
    --------
    ndarray, shape (rows * cols, n_features)
        Веса нейронов после обучения.
    """
    rows = max(1, int(params["rows"]))
    cols = max(1, int(params["cols"]))
    conn_type = _CONNECTIONS.get(str(params["conn_type"]), type_conn.grid_eight)

    parameters = som_parameters()
    parameters.init_type = _INITIALIZATIONS.get(str(params["init_type"]), type_init.uniform_grid)
    parameters.init_learn_rate = float(params["learn_rate"])
    parameters.random_state = int(params["random_state"])

    data = features.tolist()
    epochs = max(1, int(params["epochs"]))

    try:
        network = som(rows, cols, conn_type, parameters, ccore=bool(params["ccore"]))
        network.train(data, epochs)
    except OSError:
        network = som(rows, cols, conn_type, parameters, ccore=False)
        network.train(data, epochs)

    return np.asarray(network.weights, dtype=np.float64)


def _assign_to_neurons(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Относит каждую точку к ближайшему нейрону и нумерует кластеры подряд.

    Нейроны без точек пропускаются, поэтому метки идут 0, 1, 2, ...
    без пропусков.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Точки.
    weights : ndarray, shape (n_neurons, n_features)
        Веса нейронов.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки.
    """
    distances = np.linalg.norm(features[:, None, :] - weights[None, :, :], axis=2)
    winners = np.argmin(distances, axis=1)
    _, labels = np.unique(winners, return_inverse=True)
    return labels.astype(np.intp)


def _fit_predict(features: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
    """
    Обучает SOM на точках и возвращает метки.

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
    if len(features) == 0:
        return np.array([], dtype=np.intp)
    if len(features) == 1:
        return np.array([0], dtype=np.intp)

    weights = _train_network(features, params)
    return _assign_to_neurons(features, weights)


@StrategiesManager.registerStrategy(
    "som_pyc",
    "SOM (PyClustering)",
    "Самоорганизующаяся карта Кохонена из pyclustering"
)
class ConcreteStrategySOM_from_PYCLUSTERING(Strategy):
    """
    Стратегия кластеризации самоорганизующейся картой Кохонена (SOM).

    Parameters:
    -----------
    rows : int, default=1
        Число строк сетки нейронов.

    cols : int, default=3
        Число столбцов сетки нейронов.
        Максимальное число кластеров равно rows * cols.

    conn_type : {'grid_eight', 'grid_four', 'honeycomb', 'func_neighbor'}, default='grid_eight'
        Способ соединения нейронов (кто считается соседом по сетке).

    init_type : {'uniform_grid', 'random', 'random_centroid', 'random_surface'}, default='uniform_grid'
        Начальное расположение весов нейронов.

    epochs : int, default=100
        Количество эпох обучения.

    learn_rate : float, default=0.1
        Начальная скорость обучения.

    random_state : int, default=42
        Инициализация генератора случайных чисел.

    normalize : bool, default=True
        Стандартизировать признаки перед кластеризацией.

    ccore : bool, default=True
        Использовать C++ часть pyclustering (с откатом на Python).
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров, отображаемых в GUI."""
        cls._addParam(
            "rows",
            "Строк в сетке нейронов",
            StrategyParamType.UNumber,
            """
            Число строк сетки нейронов. Максимальное число кластеров
            равно rows * cols. Для обычной кластеризации удобно rows = 1,
            а число кластеров задавать через cols.
            """,
            1
        )

        cls._addParam(
            "cols",
            "Столбцов в сетке нейронов",
            StrategyParamType.UNumber,
            """
            Число столбцов сетки нейронов. Максимальное число кластеров
            равно rows * cols.
            """,
            3
        )

        cls._addParam(
            "conn_type",
            "Тип соединения нейронов",
            StrategyParamType.Switch,
            """
            Какие нейроны сетки считаются соседями:

            - grid_eight: 8 соседей (по сторонам и диагоналям)
            - grid_four: 4 соседа (только по сторонам)
            - honeycomb: 6 соседей (шестиугольная сетка)
            - func_neighbor: соседство по функции расстояния
            """,
            "grid_eight",
            switches=["grid_eight", "grid_four", "honeycomb", "func_neighbor"]
        )

        cls._addParam(
            "init_type",
            "Инициализация весов",
            StrategyParamType.Switch,
            """
            Начальное расположение весов нейронов:

            - uniform_grid: равномерная сетка в области данных (рекомендуется)
            - random: случайно в области [0, 1]
            - random_centroid: случайно около центра данных
            - random_surface: случайно в пределах области данных
            """,
            "uniform_grid",
            switches=["uniform_grid", "random", "random_centroid", "random_surface"]
        )

        cls._addParam(
            "epochs",
            "Количество эпох",
            StrategyParamType.UNumber,
            """
            Сколько раз сеть проходит по всем данным при обучении.
            Рекомендуется: 100.
            """,
            100
        )

        cls._addParam(
            "learn_rate",
            "Скорость обучения",
            StrategyParamType.UFloating,
            """
            Начальная скорость обучения: насколько сильно веса
            нейрона-победителя сдвигаются к точке. Уменьшается с каждой эпохой.
            """,
            0.1
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
            Рекомендуется оставить включённым для признаков разного масштаба.
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
        Кластеризация изображения самоорганизующейся картой.

        При числе пикселей больше _IMAGE_MAX_SAMPLES сеть обучается
        на равномерной выборке, затем все пиксели относятся
        к ближайшему нейрону.

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

        if len(pixels_proc) > _IMAGE_MAX_SAMPLES:
            step = max(1, len(pixels_proc) // _IMAGE_MAX_SAMPLES)
            sample = pixels_proc[::step][:_IMAGE_MAX_SAMPLES]
            weights = _train_network(sample, params)
            return _assign_to_neurons(pixels_proc, weights)

        return _fit_predict(pixels_proc, params)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек самоорганизующейся картой.

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
