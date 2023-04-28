import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog
from glob import glob

def find_common_edge(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    if not matches:
        return None

    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 20:
        matches = matches[:20]

    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return M


def stitch_images(images):
    stitched_image = images[0]

    for i in range(1, len(images)):
        img1 = stitched_image
        img2 = images[i]

        M = find_common_edge(img1, img2)
        if M is not None:
            h, w, _ = img2.shape
            stitched_image = cv2.warpPerspective(img1, M, (w, h))
            stitched_image = cv2.addWeighted(stitched_image, 0.5, img2, 0.5, 0)

    return stitched_image

def main():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory()

    if not folder:
        print("No folder selected.")
        sys.exit()

    image_files = sorted(glob(os.path.join(folder, '*.jpg')))
    images = [cv2.imread(image) for image in image_files]

    stitched_image = stitch_images(images)
    cv2.imwrite(os.path.join(folder, 'stitched_image.jpg'), stitched_image)

    print("Stitched image saved in the same folder as 'stitched_image.jpg'")

if __name__ == '__main__':
    main()
