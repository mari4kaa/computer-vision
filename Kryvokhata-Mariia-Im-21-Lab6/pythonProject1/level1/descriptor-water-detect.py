import cv2
import numpy as np
import imutils

def color_correction(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    # Color correction - simple histogram equalization in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    corrected_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return corrected_image

def color_clustering(image, lower_water, upper_water):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_water, upper_water)
    img_bitwise_and = cv2.bitwise_and(image, image, mask=mask)
    # Median blurring
    img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)
    return mask, img_median_blurred

def find_contours(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def identify_objects(image, contours):
    identified_image = image.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small areas
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                cv2.circle(identified_image, (cX, cY), 5, (0, 255, 0), -1)  # Draw circle at center
    return identified_image

def harris_corner_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 20, 9, 0.0000001)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red
    return image

def sift_descriptors_on_harris(image):
    # Create SIFT detector
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des

def sift_feature_matching(kp1, des1, kp2, des2):
    # Create a FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

def draw_matches(img1, kp1, img2, kp2, matches):
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

def process_images(operational_image_path, high_res_image_path):
    # Color correction
    operational_image = color_correction(operational_image_path)
    high_res_image = color_correction(high_res_image_path)

    # For two different bing pictures
    # bing_picture.png
    op_lower_water = np.array([55, 150, 11])
    op_upper_water = np.array([100, 255, 255])
    # bing_picture2.png
    hg_lower_water = np.array([55, 150, 11])
    hg_upper_water = np.array([100, 255, 255])

    # For landsat_picture and bing_picture
    # landsat_picture.png
    # op_lower_water = np.array([25, 0, 0])
    # op_upper_water = np.array([240, 255, 30])
    # bing_picture.png
    # hg_lower_water = np.array([55, 150, 11])
    # hg_upper_water = np.array([100, 255, 255])

    # Color clustering
    operational_mask, op_img_median_blurred = color_clustering(operational_image, op_lower_water, op_upper_water)
    high_res_mask, hg_img_median_blurred = color_clustering(high_res_image, hg_lower_water, hg_upper_water)

    # Find contours
    operational_contours = find_contours(operational_image, operational_mask)
    high_res_contours = find_contours(high_res_image, high_res_mask)

    # Identifying objects
    operational_ident_image = identify_objects(operational_image, operational_contours)
    high_res_ident_image = identify_objects(high_res_image, high_res_contours)

    # Apply Harris corner detection
    operational_harris_image = harris_corner_detector(op_img_median_blurred)
    high_res_harris_image = harris_corner_detector(hg_img_median_blurred)

    # Get SIFT descriptors based on Harris corners
    kp1, des1 = sift_descriptors_on_harris(operational_harris_image)
    kp2, des2 = sift_descriptors_on_harris(high_res_harris_image)

    # Perform SIFT feature matching
    good_matches = sift_feature_matching(kp1, des1, kp2, des2)

    # Draw keypoints on the images
    for kp in kp1:
        cv2.circle(operational_ident_image, (int(kp.pt[0]), int(kp.pt[1])), 5, (0, 255, 0), -1)  # Green circles
    for kp in kp2:
        cv2.circle(high_res_ident_image, (int(kp.pt[0]), int(kp.pt[1])), 5, (0, 255, 0), -1)  # Green circles

    # Resize images to have the same height for side-by-side display
    height1, width1 = operational_ident_image.shape[:2]
    height2, width2 = high_res_ident_image.shape[:2]
    new_height = min(height1, height2)

    operational_ident_resized = imutils.resize(operational_ident_image, height=new_height)
    high_res_ident_resized = imutils.resize(high_res_ident_image, height=new_height)

    # Create side by side display
    combined_image = np.hstack((operational_ident_resized, high_res_ident_resized))

    # Draw matches
    if good_matches:
        matched_image = draw_matches(op_img_median_blurred, kp1, hg_img_median_blurred, kp2, good_matches)
    else:
        matched_image = np.zeros((600, 1200, 3), dtype=np.uint8)  # Create an empty image if no matches found

    # Draw a white line in the middle to separate the two images
    line_position = operational_ident_resized.shape[1]  # Position of the line
    cv2.line(combined_image, (line_position, 0), (line_position, new_height), (255, 255, 255), 2)

    # Calculate probability of identification
    total_descriptors = min(len(kp1), len(kp2))
    match_count = len(good_matches)
    probability = (match_count / total_descriptors) * 100 if total_descriptors > 0 else 0

    print(f'Number of matches: {match_count}')
    print(f'Probability of identification: {probability:.2f}%')

    # Show results
    cv2.imshow("All Keypoints", combined_image)
    cv2.imshow("Matched Keypoints", matched_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    operational_image_path = 'landsat_picture.png'
    high_res_image_path = 'bing_picture.png'
    process_images(operational_image_path, high_res_image_path)
