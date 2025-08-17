import cv2
import numpy as np
import imutils


def color_correction(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    # color correction - simple histogram equalization in LAB color space
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
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image, contours


def identify_objects(image, contours):
    identified_image = image.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small areas
            cv2.drawContours(identified_image, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                cv2.putText(identified_image, 'Water', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return identified_image


def show_results(original, segment, contour, identified, img_type):
    original_text = 'Color corrected Input ' + img_type + ' Image'
    segment_text = 'Color Clustering ' + img_type + ' Image'
    contour_text = 'Contour ' + img_type + ' Image'
    identified_text = 'Identified ' + img_type + ' Image'

    cv2.imshow(original_text, original)
    cv2.imshow(segment_text, segment)
    cv2.imshow(contour_text, contour)
    cv2.imshow(identified_text, identified)


def process_images(operational_image_path, high_res_image_path):
    # Color correction
    operational_image = color_correction(operational_image_path)
    high_res_image = color_correction(high_res_image_path)

    #hg_lower_water = np.array([55, 150, 11])
    #hg_upper_water = np.array([120, 255, 255])
    hg_lower_water = np.array([62, 150, 11])
    hg_upper_water = np.array([100, 255, 255])

    op_lower_water = np.array([25, 0, 0])
    op_upper_water = np.array([240, 255, 30])

    # Color clustering
    operational_mask, op_img_median_blurred = color_clustering(operational_image, op_lower_water, op_upper_water)
    high_res_mask, hg_img_median_blurred = color_clustering(high_res_image,  hg_lower_water, hg_upper_water)

    # Contours
    operational_contour_image, operational_contours = find_contours(operational_image, operational_mask)
    high_res_contour_image, high_res_contours = find_contours(high_res_image, high_res_mask)

    # Identifying objects
    operational_ident_image = identify_objects(operational_image, operational_contours)
    high_res_ident_image = identify_objects(high_res_image, high_res_contours)

    #show_results(operational_image, op_img_median_blurred, operational_contour_image, operational_ident_image, 'Operational')
    show_results(high_res_image, hg_img_median_blurred, high_res_contour_image, high_res_ident_image, 'High Resolution')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    operational_image_path = 'landsat_picture.png'
    high_res_image_path = 'bing_picture.png'
    process_images(operational_image_path, high_res_image_path)
