import numpy as np
from scipy.stats import pearsonr


def euclidean_distance(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: a float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        return np.sqrt(np.sum(np.square(x - y)))
    else:
        return np.sqrt(np.sum(np.square(x - y)))


def manhattan_distance(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: a float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        return np.sum(np.abs(x - y))
    else:
        return np.sum(np.abs(x - y))


def minkowski_distance(x, y, p):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :param p: power of the distance formula
        When p=1 it's Manhattan Distance, when p=2 it's Euclidean Distance.
    :return: a float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        return np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)
    else:
        return np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)


def chebyshev_distance(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: a float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        return np.max(np.abs(x - y))
    else:
        return np.max(np.abs(x - y))


def hamming_distance(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: an int number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    else:
        x = str(x)
        y = str(y)
        d = 0
        for i in range(len(x)):
            if x[i] != y[i]:
                d += 1
        return d


def mahalanobis_distance(x, y):
    pass


def jaccard_similarity(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: an float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        return 1 - len(np.intersect1d(x, y)) / len(np.union1d(x, y))
    else:
        return 1 - len(np.intersect1d(x, y)) / len(np.union1d(x, y))


def cosine_similarity(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: a float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        num = float(np.dot(x, y.T))
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom
    else:
        num = float(np.dot(x, y.T))
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom


def pearson_coefficient(x, y):
    '''
    :param x: n-dims vector numpy.array or list
    :param y: n-dims vector numpy.array or list, same as x
    :return: a float number of distance
    '''
    if len(x) != len(y):
        print('Input error!')
        raise ValueError
    elif isinstance(x, list) and isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
        return pearsonr(x, y)[0]
    else:
        return pearsonr(x, y)[0]
