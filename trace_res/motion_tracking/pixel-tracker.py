import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees

#Define the select_pixel() function, which is a callback function for handling mouse events when selecting points.
def select_pixel(event, x, y, flags, param):
    global selected_points, selected_count
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        selected_count += 1

def main():
    global selected_points, selected_count

    #Create a Tkinter root window and hide it, as it's only used for the file dialog.
    root = tk.Tk()
    root.withdraw()

    #Open a file dialog to select the input video file.
    video_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if not video_file:
        print("No video file selected.")
        return
    
    #Open the video file using OpenCV's VideoCapture and read first frame.
    cap = cv2.VideoCapture(video_file)
    _, first_frame = cap.read()

    selected_points = []
    selected_count = 0

    #Create a named OpenCV window for selecting points and set the mouse callback function.
    cv2.namedWindow("Select two points: center of rotation and rotating point")
    cv2.setMouseCallback("Select two points: center of rotation and rotating point", select_pixel)

    #Display the first frame until two points are selected.
    while selected_count < 2:
        cv2.imshow("Select two points: center of rotation and rotating point", first_frame)
        cv2.waitKey(20)

    #Assign the selected points to center and rotating_point variables, and close the OpenCV window.
    center, rotating_point = selected_points
    cv2.destroyAllWindows()

    #Calculate the radius based on the selected points.
    radius = sqrt((center[0] - rotating_point[0])**2 + (center[1] - rotating_point[1])**2)
    
    distances = []
    angle_changes = []
    
    #Convert the first frame to grayscale and initialize the prev_point and prev_angle variables.
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_point = np.array([[[float(rotating_point[0]), float(rotating_point[1])]]], dtype=np.float32)
    prev_angle = atan2(rotating_point[1] - center[1], rotating_point[0] - center[0])

    #Create an empty motion history image.
    motion_history = np.zeros_like(first_frame)

    while True:
        #Read the next frame from the video.
        ret, frame = cap.read()
        if not ret:
            break
    
        #Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Calculate the optical flow between the previous and current frames using OpenCV's 'calcOpticalFlowPyrLK()'.
        next_point, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point, None)
        x1, y1 = prev_point.ravel()
        x2, y2 = next_point.ravel()
        
        #Draw a line on the motion history image to represent the motion.
        cv2.line(motion_history, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        #Add the motion history to the current frame.
        frame_with_history = cv2.add(frame, motion_history)

        #Draw circles for the center of rotation and the rotating point on the frame.
        cv2.circle(frame_with_history, center, 5, (255, 0, 0), -1)
        cv2.circle(frame_with_history, (int(x2), int(y2)), 5, (0, 255, 0), -1)

        #Calculate the distance and angle change between the previous and current points.
        distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        next_angle = atan2(y2 - center[1], x2 - center[0])
        angle_change = degrees(next_angle - prev_angle)

        #Append the distance and angle change to the corresponding lists.
        distances.append(distance)
        angle_changes.append(angle_change)

        #Display the frame with motion history.
        cv2.imshow("Motion tracking", frame_with_history)
        
        #Break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Update the previous frame, previous point, and previous angle variables.
        prev_gray = gray.copy()
        prev_point = next_point.reshape(-1, 1, 2)
        prev_angle = next_angle

    #Release the video file and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()

    #Calculate the total distance and total angle change.
    total_distance = sum(distances)
    total_angle_change = sum(angle_changes)
    
    #Print the radius, total distance, and total angle change.
    print(f"Radius: {radius:.2f} pixels")
    print(f"Total distance traveled by the pixel: {total_distance:.2f} pixels")
    print(f"Total angle traveled by the pixel: {total_angle_change:.2f} degrees")

if __name__ == "__main__":
    main()

