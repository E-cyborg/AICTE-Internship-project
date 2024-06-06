import cv2
import numpy as np

def preprocess_image(image):
  
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
  
    # Detect edges using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
  
    # Define a region of interest (ROI) for lane detection
    height, width = image.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.6 * height)),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)
    # Apply the ROI mask to the detected edges
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def hough_transform(edges):
  
    # Apply Hough Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    return lines

def draw_lines(image, lines):
  
    # Create an image with detected lines drawn on it
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

          
            # Draw detected lines on the line image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def process_frame(frame):
    # Process each frame for lane detection
    preprocessed = preprocess_image(frame)
    lines = hough_transform(preprocessed)
    line_image = draw_lines(frame, lines)
  
    # Combine the original frame with the detected lane lines
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined

# Open the video capture
cap = cv2.VideoCapture(0)  # You can specify the video file path here instead of 0 for webcam

while True:
  
    # Capture frames continuously
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame for lane detection
    result = process_frame(frame)
    
    # Display the result in real-time
    cv2.imshow('Lane Detection Result', result)
    
    # Check for key press to exit ('q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
