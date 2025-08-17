import cv2
import numpy as np

def create_road_mask(img):
    height, width = img.shape[:2]
    road_mask = np.zeros((height, width), dtype=np.uint8)

    pts = np.array([
        [0, int(height * 0.8)],                 # Near bottom-left corner
        [0, height],                            # Bottom-left corner
        [width, height],                        # Bottom-right corner
        [width, int(height * 0.55)],             # Near bottom-right corner
        [int(width * 0.6), int(height * 0.01)],  # Near top-right
        [int(width * 0.5), int(height * 0.01)]   # Near top-left
    ], np.int32)

    # Filling the mask with the trapezoidal region
    cv2.fillPoly(road_mask, [pts], 255)

    masked_img = cv2.bitwise_and(img, img, mask=road_mask)

    return masked_img, road_mask


def histogram_based_filtering(img):
    # Converting to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV ranges for vehicle detection
    lower_color1 = np.array([0, 60, 50])  # Light colors
    upper_color1 = np.array([180, 255, 255])

    lower_color2 = np.array([0, 0, 0])  # Dark colors
    upper_color2 = np.array([180, 255, 50])  # Lower V value for darker colors

    # Creating masks using defined color ranges
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)

    # Combining masks
    color_mask = cv2.bitwise_or(mask1, mask2)

    # Combining with histogram filtering
    binary = cv2.adaptiveThreshold(
        color_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 25
    )

    # Morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    filtered = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

    return filtered


def detect_vehicles(original_img, processed_img, road_mask):
    contours, _ = cv2.findContours(
        processed_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Checking if the contour is within the road mask
        if np.all(road_mask[y:y + h, x:x + w] > 0):
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:
                vehicles.append(contour)

    # Drawing detected vehicles
    result = original_img.copy()
    cv2.drawContours(result, vehicles, -1, (0, 255, 0), 2)

    # Drawing rectangles to better see detected vehicles
    for contour in vehicles:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return result


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if there are no more frames

        masked_img, road_mask = create_road_mask(frame)
        filtered = histogram_based_filtering(masked_img)
        result = detect_vehicles(frame, filtered, road_mask)

        # Display the original frame with detected vehicles
        cv2.imshow('Detected Vehicles', result)

        # Display the filtered image
        cv2.imshow('Filtered Video', filtered)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = 'highway_video.mp4'
    process_video(video_path)


if __name__ == "__main__":
    main()
