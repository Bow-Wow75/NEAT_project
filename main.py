from moviepy.editor import VideoFileClip
from yolo_pipeline import *
import cv2
import pickle
#from calibration import load_calibration

def load_calibration(calib_file):
    """

    :param calib_file:
    :return: mtx and dist
    """
    with open(calib_file, 'rb') as file:
        # print('load calibration data')
        data= pickle.load(file)
        mtx = data['mtx']       # calibration matrix
        dist = data['dist']     # distortion coefficients

    return mtx, dist

def pipeline_yolo(img):
    input_scale = 1
    calib_file = 'calibration_pickle.p'
    mtx, dist = load_calibration(calib_file)
    img_undist_ = cv2.undistort(img, mtx, dist, None, mtx)
    img_undist = cv2.resize(img_undist_, (0,0), fx=1/input_scale, fy=1/input_scale)
    #print(img_undist)
    output = vehicle_detection_yolo(img_undist)
    

    return output


if __name__ == "__main__":
    # YOLO Pipeline
    video_output = 'ped_new_2_out.mp4'
    clip1 = VideoFileClip("ped_new_2.MP4").subclip(0,3)
    clip = clip1.fl_image(pipeline_yolo)
    clip.write_videofile(video_output, audio=False)

