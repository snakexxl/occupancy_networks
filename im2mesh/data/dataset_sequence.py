from typing import Dict, List
import os
import numpy as np
from PIL import Image

from im2mesh.data.cdf_reading import read_cdf
from im2mesh.data.intersection_over_union import silhouette_to_prediction_function


def generateRandomPoints(number_of_points, silhouette_gt):
    height, width = silhouette_gt.shape
    random_height = np.random.uniform(low=0, high=height, size=number_of_points).astype(np.float32)
    random_width = np.random.uniform(0, width, number_of_points).astype(np.float32)
    random_points_in_image = np.stack([random_height, random_width], axis=1)
    #random_points_in_image = np.float32(random_points_in_image)
    return random_points_in_image


def importKeypointsFromCdf(directory='/home/johannesselbert/Documents/GitHub/inputs/cdffile'):
    list_of_key_points = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.endswith(".cdf"):
            keypoints = read_cdf(filepath) # shape ( number_frames x number_keypoints)
            list_of_key_points.append(keypoints)
            continue
        else:  # print(filename)
            continue
    list_of_key_points_concatenated = np.concatenate(list_of_key_points,0)
    #todo make it possible two use more than one cdf matrix


    return list_of_key_points_concatenated # shape ((number_frames*numbervideos) x number_keypoints)

def silhouette_gt_from_image(frame_index:int):
    """

    Args:
        frame_index:

    Returns:

    """
    image_path = f"/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe/Discussion/{frame_index}.png"
    img = Image.open(image_path)
    silhouette_gt = np.array(img)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    silhouette_gt = np.dot(silhouette_gt[..., :3], rgb_weights)
    silhouette_gt = silhouette_gt > 0
    silhouette_gt = np.float32(silhouette_gt)
    return silhouette_gt


class DatasetSilhouetteKeypoints:
    def __init__(self,validation:bool):
        self.validation = validation
        self.keypoints = importKeypointsFromCdf()
        print(validation)

    #todo #load keypoints from cdf files
    # load groundtruth Silhouette from images
    # is_in_silhoutte needs to be defined depended on the ground truth

    def __getitem__(self, index)->Dict[str, np.ndarray]:
        silhouette_gt = silhouette_gt_from_image(index)
        random_points = generateRandomPoints(1024,silhouette_gt)
        random_points_iou = generateRandomPoints(8000,silhouette_gt)
        is_in_silhoutte = silhouette_to_prediction_function(silhouette_gt)
        points_occ = np.stack([is_in_silhoutte(point)for point in random_points])
        points_iou_occ = np.stack([is_in_silhoutte(point)for point in random_points_iou])
        item= {
            'points': random_points,  #shape 999,2 im gesamten raum random generierte punkte
            'points.occ': points_occ,  # datatyp:bool,shape:999, ob der generierte Punkt innerhalb der Silhoutte ist für jeden der 2048 generierten Punkte
            'inputs': self.keypoints[index], #keypoints of the person
            #'inputs.normals': dontknow[dk]
        }
        if self.validation:
            #print('enter validation')
            item['voxels'] = silhouette_gt
            item['points_iou'] = random_points_iou
            item['points_iou.occ'] = points_iou_occ
            item['idx'] = index
        else:
            #print('did not enter')
            pass
        return item

    def __len__(self):
        return len(self.keypoints)


if __name__ == "__main__":
    keypoints = importKeypointsFromCdf()
    dataset = DatasetSilhouetteKeypoints(keypoints = keypoints)
    print(dataset[0])
    print(dataset[50])
    print(dataset[1])