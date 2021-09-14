import cdflib
import numpy as np



def read_cdf(filepath)->np.ndarray:
    """

    Args:
        filepath:

    Returns:
        keypoints : shape: ( number_frames x number_keypoints x dimension)
        e.g. keypoints: shape: 1383 x 32 x 2
    """
    cdf_file = cdflib.CDF(filepath)
    #info = cdf_file.cdf_info()
    a = cdf_file.varget(variable='Pose')
    a = a[0] # a: shape:( 1383 x 64)
    keypoints_x = a[:, 0:63:2] #shape 1383 x 32
    keypoints_y = a[:, 1:64:2] #shape 1383 x 32
    keypoints = np.stack([keypoints_x, keypoints_y], axis = 2)
    return keypoints

