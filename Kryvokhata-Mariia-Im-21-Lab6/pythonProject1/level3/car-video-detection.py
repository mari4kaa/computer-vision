import time

import cv2
import numpy as np

def create_road_mask(img):
    height, width = img.shape[:2]
    road_mask = np.zeros((height, width), dtype=np.uint8)

    pts = np.array([
        [0, int(height * 0.8)],                 # Near bottom-left corner
        [0, height],                            # Bottom-left corner
        [width, height],                        # Bottom-right corner
        [width, int(height * 0.6)],             # Near bottom-right corner
        [int(width * 0.6), int(height * 0.01)],  # Near top-right
        [int(width * 0.5), int(height * 0.01)]   # Near top-left
    ], np.int32)

    # Filling the mask with the trapezoidal region
    cv2.fillPoly(road_mask, [pts], 255)

    masked_img = cv2.bitwise_and(img, img, mask=road_mask)

    return masked_img, road_mask


if __name__ == "__main__":
    # Load the Haar cascade file
    car_cascade = cv2.CascadeClassifier('cars.xml')

    # Check if the cascade file has been loaded correctly
    if car_cascade.empty():
        raise IOError('Unable to load the car cascade classifier xml file')

    cap = cv2.VideoCapture('highway_video.mp4')

    # Define the scaling factor
    scaling_factor = 0.5

    # Iterate until the user hits the 'Esc' key
    while True:
        time.sleep(0.03)
        # Capture the current frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames

        # Resize the frame
        frame = cv2.resize(frame, None,
                           fx=scaling_factor, fy=scaling_factor,
                           interpolation=cv2.INTER_AREA)

        # Create a road mask for the current frame
        masked_img, road_mask = create_road_mask(frame)

        # Convert the masked image to grayscale
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # Run the car detector on the grayscale image
        car_rects = car_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the detected cars
        for (x, y, w, h) in car_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('Car Detector', frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(1)
        if c == 27:  # Esc key
            break

    cap.release()

    # Close all the windows
    cv2.destroyAllWindows()
