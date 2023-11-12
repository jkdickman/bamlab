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

    cv2.namedWindow("Select three points: center of rotation, first rotating point and second rotating point")
    cv2.setMouseCallback("Select three points: center of rotation, first rotating point and second rotating point", select_pixel)

    while selected_count < 3:
        cv2.imshow("Select three points: center of rotation, first rotating point and second rotating point", first_frame)
        cv2.waitKey(20)

    center, rotating_point1, rotating_point2 = selected_points[0], selected_points[1], selected_points[2]
    cv2.destroyAllWindows()

    radius1 = sqrt((center[0] - rotating_point1[0])**2 + (center[1] - rotating_point1[1])**2)
    radius2 = sqrt((center[0] - rotating_point2[0])**2 + (center[1] - rotating_point2[1])**2)
    
    distances1, distances2, angle_changes1, angle_changes2 = [], [], [], []
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_point1 = np.array([[[float(rotating_point1[0]), float(rotating_point1[1])]]], dtype=np.float32)
    prev_point2 = np.array([[[float(rotating_point2[0]), float(rotating_point2[1])]]], dtype=np.float32)
    prev_angle1 = atan2(rotating_point1[1] - center[1], rotating_point1[0] - center[0])
    prev_angle2 = atan2(rotating_point2[1] - center[1], rotating_point2[0] - center[0])

    motion_history = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_point1, status1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point1, None)
        next_point2, status2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point2, None)
        
        x11, y11 = prev_point1.ravel()
        x21, y21 = next_point1.ravel()
        x12, y12 = prev_point2.ravel()
        x22, y22 = next_point2.ravel()
        
        cv2.line(motion_history, (int(x11), int(y11)), (int(x21), int(y21)), (0, 255, 0), 2)
        cv2.line(motion_history, (int(x12), int(y12)), (int(x22), int(y22)), (0, 255, 0), 2)
        
        frame_with_history = cv2.add(frame, motion_history)

        cv2.circle(frame_with_history, center, 5, (255, 0, 0), -1)
        cv2.circle(frame_with_history, (int(x21), int(y21)), 5, (0, 255, 0), -1)
        cv2.circle(frame_with_history, (int(x22), int(y22)), 5, (0, 0, 255), -1)

        distance1 = sqrt((x21 - x11)**2 + (y21 - y11)**2)
        next_angle1 = atan2(y21 - center[1], x21 - center[0])
        angle_change1 = degrees(next_angle1 - prev_angle1)
        
        distance2 = sqrt((x22 - x12)**2 + (y22 - y12)**2)
        next_angle2 = atan2(y22 - center[1], x22 - center[0])
        angle_change2 = degrees(next_angle2 - prev_angle2)

        distances1.append(distance1)
        angle_changes1.append(angle_change1)

        distances2.append(distance2)
        angle_changes2.append(angle_change2)

        cv2.imshow("Motion tracking", frame_with_history)
  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray.copy()
        prev_point1 = next_point1.reshape(-1, 1, 2)
        prev_point2 = next_point2.reshape(-1, 1, 2)
        prev_angle1 = next_angle1
        prev_angle2 = next_angle2

    cap.release()
    cv2.destroyAllWindows()

    total_distance1 = sum(distances1)
    total_distance2 = sum(distances2)
    total_angle_change1 = sum(angle_changes1)
    total_angle_change2 = sum(angle_changes2)
    
    print(f"Radius of first point: {radius1:.2f} pixels")
    print(f"Total distance traveled by the first pixel: {total_distance1:.2f} pixels")
    print(f"Total angle traveled by the first pixel: {total_angle_change1:.2f} degrees")

    print(f"Radius of second point: {radius2:.2f} pixels")
    print(f"Total distance traveled by the second pixel: {total_distance2:.2f} pixels")
    print(f"Total angle traveled by the second pixel: {total_angle_change2:.2f} degrees")

if __name__ == "__main__":
    main()
