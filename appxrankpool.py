import os
import argparse
import cv2
import numpy as np


class Queue():
    def __init__(self, size):
        self.size = size
        self.container = []

    def enqueue(self, item):
        if len(self.container) < self.size:
            self.container.append(item)
        else:
            print('Buffer full')

    def dequeue(self):
        if not self.isempty():
            self.container.pop(0)
        else:
            print("Buffer empty")

    def get(self):
        return np.array(self.container)

    def isempty(self):
        return len(self.container) == 0

    def isfull(self):
        return (len(self.container) == self.size)


def cvApproxRankPooling_DIN(imgs):
    T = len(imgs)
  
    harmonics = []
    harmonic = 0
    for t in range(0, T+1):
        harmonics.append(harmonic)
        harmonic += float(1)/(t+1)

    weights = []
    for t in range(1 ,T+1):
        weight = 2 * (T - t + 1) - (T+1) * (harmonics[T] - harmonics[t-1])
        weights.append(weight)
        
    feature_vectors = []
    for i in range(len(weights)):
        feature_vectors.append(imgs[i] * weights[i])

    feature_vectors = np.array(feature_vectors)

    rank_pooled = np.sum(feature_vectors, axis=0)
    rank_pooled = cv2.normalize(rank_pooled, None, alpha=0, beta=255.0, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return rank_pooled


def main_vid(vid_url):
    """Main function.
    """
    cap = cv2.VideoCapture(vid_url)
    buffer = Queue(10)

    while True:
        ret, frame = cap.read()

        if buffer.isfull():
            buffer.dequeue()
            buffer.enqueue(frame)
        else:
            buffer.enqueue(frame)

        try:
            frames = buffer.get()
            rank_pooled = cvApproxRankPooling_DIN(frames)
        except TypeError:
            break

        cv2.imshow('frame', rank_pooled)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_rgb(imgs_dir, out_img):
    """Main function
    """
    fpaths = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]
    fpaths.sort()
    frames = [cv2.imread(fpath) for fpath in fpaths]

    rank_pooled = cvApproxRankPooling_DIN(frames)
    cv2.imwrite(out_img, rank_pooled)


if __name__ == '__main__':
    # default dir
    proj_dir = os.path.dirname(os.path.abspath(__file__))

    imgs_dir = os.path.join(proj_dir, "data/src/") # images folder
    
    vid_dir = os.path.join(proj_dir, "data/vid_src/")
    vid_url = os.path.join(vid_dir, "#437_How_To_Ride_A_Bike_ride_bike_f_cm_np1_ba_med_0.avi") # video file

    # parser
    parser = argparse.ArgumentParser(description="Welcome to approximated rank pooling demo.")
    parser.add_argument('-s', '--source', type=str, default=vid_url,
            help="Video file or Images folder path")
    parser.add_argument('-d', '--dest', type=str, required=True, help="Folder/dir to save output")
    args = parser.parse_args()

    src = args.source
    dest = args.dest
    
    if os.path.isfile(src): # if it's a file, we assume it's a video
        main_vid(src)
    elif os.path.isdir(src):
        out_img = os.path.join(dest, 'py_rprgb_00000.jpg')
        main_rgb(src, out_img)

