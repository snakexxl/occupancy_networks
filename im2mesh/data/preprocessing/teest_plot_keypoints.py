from im2mesh.data.preprocessing.dump_png_from_videos import dump_png_from_videos

if __name__ == "__main__":
    #todo glob_expression should take all the videos
    glob_expression_videos = "/home/johannesselbert/Documents/GitHub/inputs/groudtruthvideosingle/Discussion.60457274.mp4"
    save_dir = "/home/johannesselbert/Documents/GitHub/inputs/groundtruthvideoframe"

    dump_png_from_videos(glob_expression_videos, save_dir)
    print("fertig")


