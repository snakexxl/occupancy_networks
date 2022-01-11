from im2mesh.data.preprocessing.dump_png_from_videos import dump_png_from_videos
import os


if __name__ == "__main__":
    #todo glob_expression should take all the videos
    glob_expression_videos_training = "/home/johannesselbert/Documents/GitHub/inputs/groudtruthvideosingle/training"
    save_dir_training = "/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe"
training = True
idx = 0
for filename in sorted(os.listdir(glob_expression_videos_training)):
    filepath = os.path.join(glob_expression_videos_training, filename)
    if filepath.endswith(".mp4"):
        idx = dump_png_from_videos(filepath, save_dir_training, idx, training)
        print("idx")
        print(filepath + "fertig")
    else:  # print(filename)
        continue
    print("fertig training")

idx = 0
training = False
glob_expression_videos_testing = "/home/johannesselbert/Documents/GitHub/inputs/groudtruthvideosingle/testing"
save_dir_testing = "/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe"
for filename in sorted(os.listdir(glob_expression_videos_testing)):
    filepath = os.path.join(glob_expression_videos_testing, filename)
    if filepath.endswith(".mp4"):
        idx = dump_png_from_videos(filepath, save_dir_testing, idx, training)
        print("idx")
        print(filepath + "fertig")
    else:  # print(filename)
        continue
    print("fertig training")





