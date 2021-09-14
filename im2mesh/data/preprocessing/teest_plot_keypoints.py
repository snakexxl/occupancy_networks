from im2mesh.data.preprocessing.dump_png_from_videos import dump_png_from_videos

if __name__ == "__main__":
    glob_expression_videos = "/home/john/Github/occupancy_networks/data/inputs/gt_video/Directions.60457274.mp4"
    save_dir = "/home/john/Github/occupancy_networks/data/inputs/gt_video_frame"

    dump_png_from_videos(glob_expression_videos, save_dir)
    print("fertig")


