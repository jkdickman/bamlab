import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog
from glob import glob

def find_common_edge(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return None

    src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return M

def stitch_images(images):
    stitched_image = images[0]

    for i in range(1, len(images)):
        img1 = stitched_image
        img2 = images[i]

        M = find_common_edge(img1, img2)
        if M is not None:
            h1, w1, _ = img1.shape
            h2, w2, _ = img2.shape

            corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)
            x_min, y_min = np.int32(transformed_corners.min(axis=0).ravel())
            x_max, y_max = np.int32(transformed_corners.max(axis=0).ravel())

            stitched_image_width = max(x_max, w1 + w2)
            img1_warped = cv2.warpPerspective(img1, M, (stitched_image_width, h1))
            stitched_image = np.zeros((h1, stitched_image_width, 3), dtype=np.uint8)
            stitched_image[:, :w1] = img1_warped
            stitched_image[:, w1:w1 + w2] = img2

            gray1 = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            _, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
            _, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
            mask = mask1 * mask2

            for y in range(h1):
                for x in range(w1, w1 + w2):
                    if mask[y, x] == 255:
                        alpha = (x - w1) / w2
                        stitched_image[y, x] = alpha * img1_warped[y, x] + (1 - alpha) * img2[y, x - w1]

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
