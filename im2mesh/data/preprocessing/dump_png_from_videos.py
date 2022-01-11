import os
from glob import glob
import cv2


def dump_png_from_videos(glob_expression: str, save_dir: str, idx: int, training:bool):
    training = training
    video_paths = glob(glob_expression)
    idx_frame = 0
    for path in video_paths:
        idx_frame= save_frame(path, save_dir, idx, training, gap=1)

    print(idx_frame)
    return idx_frame

def save_frame(video_path, save_dir, idx, training,gap=1):
    #uncomment to create a file which is named from the subjekt from human3.6
    #name = video_path.split("/")[-1].split(".")[0]
    if training:
        name = 'train'
    else:
        name = 'test'
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)



    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            return idx

        if idx == 0:
            cv2.imwrite(f"{save_path}/{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1




def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")


