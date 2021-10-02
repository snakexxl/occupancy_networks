

#import numpy as np
#import tensorflow as tf
import torch

#from im2mesh.data.dataset_sequence import importKeypointsFromCdf
torch.cuda.is_available()

#points =
keypoints = importKeypointsFromCdf(directory='/home/john/Github/occupancy_networks/data/inputs/model_testing/')
#poseMatrix = keypoints[0,: , :]
#bestModel = f'/home/john/Github/occupancy_networks/out/pointcloud/onet/model_best.pt'
#modelDic = torch.load(bestModel)
#modelWeight =
#prediction = modelDic.predict([poseMatrix])
#modelDic.model(poseMatrix)
#model = tf.keras.models.load_model(bestModel)

#prediction = model.predict([keypoints, points])
#print(prediction)
print("hello")