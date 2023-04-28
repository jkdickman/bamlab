import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees

def select_pixel(event, x, y, flags, param):
    global selected_points, selected_count
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        selected_count += 1

def main():
    global selected_points, selected_count

    root = tk.Tk()
    root.withdraw()

    video_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if not video_file:
        print("No video file selected.")
        return

    cap = cv2.VideoCapture(video_file)
    _, first_frame = cap.read()

    selected_points = []
    selected_count = 0

    cv2.namedWindow("Select two points: center of rotation and rotating point")
    cv2.setMouseCallback("Select two points: center of rotation and rotating point", select_pixel)

    while selected_count < 2:
        cv2.imshow("Select two points: center of rotation and rotating point", first_frame)
        cv2.waitKey(20)

    center, rotating_point = selected_points
    cv2.destroyAllWindows()

    radius = sqrt((center[0] - rotating_point[0])**2 + (center[1] - rotating_point[1])**2)
    
    distances = []
    angle_changes = []
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_point = np.array([[[float(rotating_point[0]), float(rotating_point[1])]]], dtype=np.float32)
    prev_angle = atan2(rotating_point[1] - center[1], rotating_point[0] - center[0])

    motion_history = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_point, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point, None)
        x1, y1 = prev_point.ravel()
        x2, y2 = next_point.ravel()

        cv2.line(motion_history, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        frame_with_history = cv2.add(frame, motion_history)

        cv2.circle(frame_with_history, center, 5, (255, 0, 0), -1)
        cv2.circle(frame_with_history, (int(x2), int(y2)), 5, (0, 255, 0), -1)

        distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        next_angle = atan2(y2 - center[1], x2 - center[0])
        angle_change = degrees(next_angle - prev_angle)

        distances.append(distance)
        angle_changes.append(angle_change)

        cv2.imshow("Motion tracking", frame_with_history)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray.copy()
        prev_point = next_point.reshape(-1, 1, 2)
        prev_angle = next_angle

    cap.release()
    cv2.destroyAllWindows()

    total_distance = sum(distances)
    total_angle_change = sum(angle_changes)
    print(f"Radius: {radius:.2f} pixels")
    print(f"Total distance traveled by the pixel: {total_distance:.2f} pixels")
    print(f"Total angle traveled by the pixel: {total_angle_change:.2f} degrees")

if __name__ == "__main__":
    main()

