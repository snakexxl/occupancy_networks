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
    # random_points_in_image = np.float32(random_points_in_image)
    return random_points_in_image


def importKeypointsFromCdf(mode):
    if mode == 'train':
        directory = '/home/johannesselbert/Documents/GitHub/inputs/cdffile/train'
    else:
        if mode == 'test':
            directory = '/home/johannesselbert/Documents/GitHub/inputs/cdffile/test'
        else:
            if mode == 'val':
                directory = '/home/johannesselbert/Documents/GitHub/inputs/cdffile/val'
            else:
                print('Error: unknown set for cdf!!! Wrong mode ')
    list_of_key_points = []
    for filename in sorted(os.listdir(directory)):
        print(mode + "cdf:")
        print(filename)
        filepath = os.path.join(directory, filename)
        if filepath.endswith(".cdf"):
            keypoints = read_cdf(filepath)  # shape ( number_frames x number_keypoints)
            list_of_key_points.append(keypoints)
            continue
        else:  # print(filename)
            continue
    list_of_key_points_concatenated = np.concatenate(list_of_key_points, 0)

    return list_of_key_points_concatenated  # shape ((number_frames*numbervideos) x number_keypoints)


def silhouette_gt_from_image(frame_index: int, mode: str):
    """

    Args:
        mode:
        frame_index:

    Returns:

    """
    if mode == 'train':
        image_path = f"/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe/train/{frame_index}.png"
    else:
        if mode == 'test':
            image_path = f"/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe/test/{frame_index}.png"
        else:
            if mode == 'val':
                #todo Pfad für validation erstellen
                image_path = f"/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe/val/{frame_index}.png"
            else:
                print('Error: kein Pfad für die Silhouette möglich')

    img = Image.open(image_path)
    silhouette_gt = np.array(img)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    silhouette_gt = np.dot(silhouette_gt[..., :3], rgb_weights)
    silhouette_gt = silhouette_gt > 0
    silhouette_gt = np.float32(silhouette_gt)
    while silhouette_gt.shape[0] > 1000:
        silhouette_gt = np.delete(silhouette_gt, 1000, 0)
    while silhouette_gt.shape[1] > 1000:
        silhouette_gt = np.delete(silhouette_gt, 1000, 1)
    return silhouette_gt


class DatasetSilhouetteKeypoints:
    # def __init__(self, validation: bool, test: bool):
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'test':
            print('test set wird gemacht')
        else:
            if self.mode == 'val':
                print('validation set wird gemacht')
            else:
                if self.mode == 'train':
                    print('train set wird gemacht')
                else:
                    print('Error: unknown set!!! Wrong mode ')

        self.keypoints = importKeypointsFromCdf(mode)
        # else:
        #     all_keypoints = importKeypointsFromCdf(mode)
        #     size_train = int(0.90 * len(all_keypoints))
        #     if self.mode == 'val':
        #         self.keypoints_validation = all_keypoints[size_train:]
        #     else:
        #         # der name keypointsvalidation ist nicht gut gewählt, weil es eigentlich keypoints von training sind. Es macht es aber leichter zwischen train und validation zu unterscheiden, ohne viel mehr code
        #         self.keypoints_validation = all_keypoints[:size_train]

    # todo #load keypoints from cdf files
    # load groundtruth Silhouette from images
    # is_in_silhoutte needs to be defined depended on the ground truth

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        #print(self.mode + '<-- Dieser mode wird ausgeführt')
        #print(index)
        silhouette_gt = silhouette_gt_from_image(index, self.mode)
        random_points = generateRandomPoints(2048, silhouette_gt)
        random_points_iou = generateRandomPoints(16000, silhouette_gt)
        is_in_silhoutte = silhouette_to_prediction_function(silhouette_gt)
        points_occ = np.stack([is_in_silhoutte(point) for point in random_points])
        points_iou_occ = np.stack([is_in_silhoutte(point) for point in random_points_iou])
        item = {
            'points': random_points,  # shape 999,2 im gesamten raum random generierte punkte
            'points.occ': points_occ,
            # datatyp:bool,shape:999, ob der generierte Punkt innerhalb der Silhoutte ist für jeden der 2048 generierten Punkte
            'inputs': self.keypoints[index],  # keypoints of the person
            # 'inputs.normals': dontknow[dk]
        }
        if self.mode == 'val' or self.mode == 'test':
            item['voxels'] = silhouette_gt
            item['points_iou'] = random_points_iou
            item['points_iou.occ'] = points_iou_occ
            item['idx'] = index
            item['original_silhouette'] = silhouette_gt
        else:
            if not self.mode == 'train':
                print('Error: Weder test,val,train Fehler')
            pass
        return item

    def __len__(self):
        return len(self.keypoints)


# if __name__ == "__main__":
# keypoints = importKeypointsFromCdf()
# dataset = DatasetSilhouetteKeypoints(keypoints = keypoints)
# print(dataset[0])
# print(dataset[50])
# print(dataset[1])
