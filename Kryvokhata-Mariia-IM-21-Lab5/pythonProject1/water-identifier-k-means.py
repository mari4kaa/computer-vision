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


def kmeans_clustering(image, K=2):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    # defining criteria and applying kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(twoDimage, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape((img.shape))

    return clustered_image, labels, centers


def create_binary_mask(clustered_image, labels, cluster_index):
    mask = np.zeros(clustered_image.shape[:2], dtype=np.uint8)
    mask[labels.reshape(clustered_image.shape[:2]) == cluster_index] = 255
    return mask


def find_water_cluster(centers, target_color):
    # calculating the Euclidean distance between each cluster center and the target color
    distances = np.linalg.norm(centers - np.array(target_color), axis=1)
    water_cluster = np.argmin(distances)  # the cluster with the smallest distance
    print(f"Water cluster index: {water_cluster}, center color: {centers[water_cluster]}")
    return water_cluster


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


def show_results(original, clustered, contour, identified, img_type):
    original_text = 'Color corrected Input ' + img_type + ' Image'
    segment_text = 'K-means Clustering ' + img_type + ' Image'
    contour_text = 'Contour ' + img_type + ' Image'
    identified_text = 'Identified ' + img_type + ' Image'

    cv2.imshow(original_text, original)
    cv2.imshow(segment_text, clustered)
    cv2.imshow(contour_text, contour)
    cv2.imshow(identified_text, identified)


def process_images(operational_image_path, high_res_image_path):
    # Color correction
    operational_image = color_correction(operational_image_path)
    high_res_image = color_correction(high_res_image_path)

    # defining water color (BGR for #08100b)
    water_color = (11, 16, 8)

    # K-means clustering
    operational_clustered, operational_labels, operational_centers = kmeans_clustering(operational_image, K=10)
    high_res_clustered, high_res_labels, high_res_centers = kmeans_clustering(high_res_image, K=20)

    operational_water_cluster = find_water_cluster(operational_centers, water_color)
    high_res_water_cluster = find_water_cluster(high_res_centers, water_color)

    # Binary masks for the water clusters
    operational_mask = create_binary_mask(operational_clustered, operational_labels, cluster_index=operational_water_cluster)
    high_res_mask = create_binary_mask(high_res_clustered, high_res_labels, cluster_index=high_res_water_cluster)

    # Contours
    operational_contour_image, operational_contours = find_contours(operational_image, operational_mask)
    high_res_contour_image, high_res_contours = find_contours(high_res_image, high_res_mask)

    # Identifying objects
    operational_ident_image = identify_objects(operational_image, operational_contours)
    high_res_ident_image = identify_objects(high_res_image, high_res_contours)

    show_results(operational_image, operational_clustered, operational_contour_image, operational_ident_image, 'Operational')
    show_results(high_res_image, high_res_clustered, high_res_contour_image, high_res_ident_image, 'High Resolution')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    operational_image_path = 'landsat_picture.png'
    high_res_image_path = 'bing_picture.png'
    process_images(operational_image_path, high_res_image_path)
