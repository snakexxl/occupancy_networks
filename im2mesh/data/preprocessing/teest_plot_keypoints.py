from im2mesh.data.preprocessing.dump_png_from_videos import dump_png_from_videos
import os


if __name__ == "__main__":
    #todo glob_expression should take all the videos
    glob_expression_videos = "/home/johannesselbert/Documents/GitHub/inputs/groudtruthvideosingle"
    save_dir = "/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe"

idx = 0
for filename in os.listdir(glob_expression_videos):
    filepath = os.path.join(glob_expression_videos, filename)
    if filepath.endswith(".mp4"):
        idx = dump_png_from_videos(filepath, save_dir, idx)
        print("idx")
        print(filepath + "fertig")
    else:  # print(filename)
        continue
    print("fertig")





