from typing import Callable

import numpy as np


def intersection_over_union_random(silhouette: np.ndarray, silhouette_gt: np.ndarray) \
        -> float:
    prediction_function = silhouette_to_prediction_function(silhouette)
    score = evaluate_prediction_function(prediction_function, silhouette_gt)
    return score


def intersection_over_union(silhouette: np.ndarray, silhouette_gt: np.ndarray) \
        -> float:
    """
    Computes the intersection score over union score between two pixel
    wise silhouette predictions.
    Args:
        silhouette: shape: (height, width)
        silhouette_gt:
            groundtruthtable of the silhouette
            shape: (height, width)

    Returns: score between 0 and 1

    """
    intersection = np.logical_and(silhouette, silhouette_gt)
    union = np.logical_or(silhouette, silhouette_gt)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def silhouette_to_prediction_function(
        silhouette: np.ndarray
) -> Callable[[np.ndarray], bool]:
    """
    Takes a silhouette and returns a function.
    The returned function takes x,y point and
     returns wether it is in the silhouette.


    Args:
        silhouette:

    Returns:

    """

    def prediction_function(point: np.ndarray) -> bool:
        return silhouette[int(point[0]), int(point[1])]

    return prediction_function


def evaluate_prediction_function(prediction_function: Callable[[np.ndarray], bool],
                                 silhouette_gt: np.ndarray,
                                 number_evaluation_points: int = 1000):
    """

    Args:
        number_evaluation_points: The number of points for evaluation
        prediction_function: Takes an x, y point and returns wether the point
                             is in the silhouette.
        silhouette_gt: shape:
            groundtruthtable of the silhouette
            shape: (height, width)

    Returns:

    """
    height, width = silhouette_gt.shape
    random_height = np.random.uniform(low=0, high=height, size=number_evaluation_points)
    random_width = np.random.uniform(0, width, number_evaluation_points)
    random_points = np.stack([random_height, random_width], axis=1)
    """prediction=[]
    for point in random_points:
        val = prediction_function(point)
        prediction.append(val)"""
    prediction_points = [prediction_function(point) for point in random_points]
    prediction_function_gt = silhouette_to_prediction_function(silhouette_gt)
    prediction_gt = [prediction_function_gt(point) for point in random_points]
    prediction_points = np.array(prediction_points)
    prediction_gt = np.array(prediction_gt)
    score = intersection_over_union(prediction_points, prediction_gt)
    return score
