import cv2
import math
import os
import numpy as np
from skimage.filters import gabor
from skimage.filters.rank import equalize
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.transform import hough_circle, hough_circle_peaks
from scipy import ndimage
import itertools

space_constant_x1 = 3
space_constant_x2 = 4.5
space_constant_y = 1.5


f1 = 0.1
f2 = 0.07
x1 = range(-9, 10, 1)
x2 = range(-14, 15, 1)
y = range(-5, 6, 1)



def quantize_using_msb(feature_vector):
    """
    Quantizes a real-valued feature vector to 3-bit binary strings using the first 3 MSBs.
    
    Args:
        feature_vector (list or np.ndarray): Array of real numbers.
    
    Returns:
        list: A list of 3-bit binary strings.
    """
    feature_vector = np.array(feature_vector)  # Ensure it's a numpy array
    
    # Scale the feature vector to integers
    max_int = 2**16  # Assume 16-bit representation
    scaled_vector = (feature_vector / feature_vector.max() * max_int).astype(int)
    
    # Convert each integer to binary and extract the first 3 MSBs
    binary_strings = [format(value, '016b')[:3] for value in scaled_vector]  # changed bit extraction here 
    return binary_strings
def FeatureExtraction(roi):
    filter1 = []
    filter2 = []
    f1 = 0.1
    f2 = 0.07
    x1 = range(-9, 10, 1)
    x2 = range(-14, 15, 1)
    y = range(-5, 6, 1)
    space_constant_x1 = 3
    space_constant_x2 = 4.5
    space_constant_y = 1.5
    for j in range(len(y)):
        for i in range(len(x1)):
            cell_1 = gabor_filter(x1[i], y[j], space_constant_x1, space_constant_y, f1,True)
            filter1.append(cell_1)
        for k in range(len(x2)):
            cell_2 = gabor_filter(x2[k], y[j], space_constant_x2, space_constant_y, f2, True)
            filter2.append(cell_2)
    filter1 = np.reshape(filter1, (len(y), len(x1)))
    filter2 = np.reshape(filter2, (len(y), len(x2)))

    filtered_eye1 = ndimage.convolve(roi, np.real(filter1), mode='wrap', cval=0)
    filtered_eye2 = ndimage.convolve(roi, np.real(filter2), mode='wrap', cval=0)
    # filtered_eye1, im2 = gabor(roi, frequency=0.1)
    # filtered_eye2, _ = gabor(roi, frequency=0.07)
    
    # real_part = np.real(filtered_eye1)
    # imag_part = np.real(im2)
    # # print(imag_part[:100])
    # # Phase quantization for both filter outputs
    # feature_vector1 = phaseQuantisation(real_part, imag_part)
    # # feature_vector2 = phaseQuantisation(filtered_eye_real2, filtered_eye_imag2)

    # # Combine both feature vectors
    # final_feature_vector = feature_vector1 
    # # print(f"Vector length: {len(final_feature_vector)}")  # 1D length
    # # print(f"Vector shape: {np.shape(final_feature_vector)}")
    # return final_feature_vector
# 32768

    vector = []
    i = 0
    while i < roi.shape[0]:
        j = 0
        while j < roi.shape[1]:
            mean1 = filtered_eye1[i:i + 8, j:j + 8].mean()
            mean2 = filtered_eye2[i:i + 8, j:j + 8].mean()
            AAD1 = abs(filtered_eye1[i:i + 8, j:j + 8] - mean1).mean()
            AAD2 = abs(filtered_eye2[i:i + 8, j:j + 8] - mean2).mean()

            vector.append(mean1)
            vector.append(AAD1)
            vector.append(mean2)
            vector.append(AAD2)
            j = j + 8
        i = i + 8
    vector = np.array(vector)
    # print(f"Vector length: {len(vector)}")  # 1D length
    # print(f"Vector shape: {np.shape(vector)}")
    # print(vector[:100])
    fv = quantize_using_msb(vector)
    # print(fv[:20])
    return fv


# 1. Segmentation - Function to detect and segment iris and pupil

def segment_iris_1(eye):
    # Convert image to grayscale
    blured = cv2.bilateralFilter(eye, 9, 100, 100)
    Xp = blured.sum(axis=0).argmin()
    Yp = blured.sum(axis=1).argmin()
    x = blured[max(Yp - 60, 0):min(Yp + 60, 280), max(Xp - 60, 0):min(Xp + 60, 320)].sum(axis=0).argmin()
    y = blured[max(Yp - 60, 0):min(Yp + 60, 280), max(Xp - 60, 0):min(Xp + 60, 320)].sum(axis=1).argmin()
    Xp = max(Xp - 60, 0) + x
    Yp = max(Yp - 60, 0) + y
    if Xp >= 100 and Yp >= 80:
        blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)
        pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12,
                                         minRadius=15, maxRadius=80)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
        xp = Xp - 60 + xp
        yp = Yp - 60 + yp
    else:
        pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 280, minRadius=25, maxRadius=55, param2=51)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
    eye_copy = eye.copy()
    rp = rp + 7  # slightly enlarge the pupil radius makes a better result
    blured_copy = cv2.medianBlur(eye_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)
    eye_edges[:, xp - rp - 30:xp + rp + 30] = 0

    hough_radii = np.arange(rp + 45, 150, 2)
    hough_res = hough_circle(eye_edges, hough_radii)
    accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    iris = []
    iris.extend(xi)
    iris.extend(yi)
    iris.extend(ri)
    if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
        iris[0] = xp
        iris[1] = yp
    return np.array(iris), np.array([xp, yp, rp])

# 2. Normalization - Function to unwrap iris from circular to rectangular (polar) form

def IrisNormalization(image, inner_circle, outer_circle):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        localized_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        localized_img = image
    localized_img = image
    row = 64
    col = 512
    normalized_iris = np.zeros(shape=(64, 512))
    inner_y = inner_circle[0]  # height
    inner_x = inner_circle[1]  # width
    outer_y = outer_circle[0]
    outer_x = outer_circle[1]
    angle = 2.0 * math.pi / col
    inner_boundary_x = np.zeros(shape=(1, col))
    inner_boundary_y = np.zeros(shape=(1, col))
    outer_boundary_x = np.zeros(shape=(1, col))
    outer_boundary_y = np.zeros(shape=(1, col))
    for j in range(col):
        inner_boundary_x[0][j] = inner_circle[0] + inner_circle[2] * math.cos(angle * (j))
        inner_boundary_y[0][j] = inner_circle[1] + inner_circle[2] * math.sin(angle * (j))

        outer_boundary_x[0][j] = outer_circle[0] + outer_circle[2] * math.cos(angle * (j))
        outer_boundary_y[0][j] = outer_circle[1] + outer_circle[2] * math.sin(angle * (j))

    for j in range(512):
        for i in range(64):
            normalized_iris[i][j] = localized_img[min(int(int(inner_boundary_y[0][j])
                                                          + (int(outer_boundary_y[0][j]) - int(
                inner_boundary_y[0][j])) * (i / 64.0)), localized_img.shape[0] - 1)][min(int(int(inner_boundary_x[0][j])
                                                                                             + (int(
                outer_boundary_x[0][j]) - int(inner_boundary_x[0][j]))
                                                                                             * (i / 64.0)),
                                                                                         localized_img.shape[1] - 1)]

    res_image = 255 - normalized_iris
    return res_image
  
# 4. Main function that processes the eye image
def process_eye_image(image_path):
    # Read the eye image
    image = cv2.imread(image_path,0)

    # Step 1: Segment the iris and pupil
    iris_circle, pupil_circle = segment_iris_1(image)
    iris, pupil = segment_iris_1(image)

# Draw the pupil circle in red
    cv2.circle(image, (pupil[0], pupil[1]), pupil[2], (255, 0, 0), 2)  # Red circle for the pupil

    # Draw the iris circle in blue with more thickness
    cv2.circle(image, (iris[0], iris[1]), iris[2], (255, 0, 0), 4)  # Blue circle for iris with 4px thickness

    # # Display the image with the segmented pupil and iris
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    # plt.title('Segmented Iris and Pupil')
    # plt.axis('off')  # Hide axis
    # plt.show()

    # Step 2: Normalize the iris to a fixed size
    normalized_iris = IrisNormalization(image, iris_circle, pupil_circle)

    # Step 3: Encode the normalized iris to extract features
    feature_vector = FeatureExtraction(normalized_iris)
    
    return feature_vector

# 5. Displaying the Feature Vector
def display_feature_vector(feature_vector):
    plt.figure(figsize=(12, 4))
    plt.plot(feature_vector)
    plt.title("Feature Vector")
    plt.xlabel("Index")
    plt.ylabel("Value")
     # Set y-axis limits for better visibility
    plt.show()

    # Display a portion of the actual array
    print("First 100 elements of the feature vector:")
    print(feature_vector[:100])

def hamming_distance_string(vector1, vector2):
    # Ensure both vectors are of the same length
    assert len(vector1) == len(vector2), "Vectors must be of the same length"
    
    # Count the differing positions
    differing_positions = sum(el1 != el2 for el1, el2 in zip(vector1, vector2))
    
    # Calculate normalized Hamming distance
    normalized_distance = differing_positions / len(vector1)
    
    return normalized_distance

def hamming_distance_3String(vector1, vector2):
    assert len(vector1) == len(vector2), "Vectors must be of the same length"
    
    # Count the differing position
    differing_positions=0
    for el1, el2 in zip(vector1, vector2):
        differing_positions += sum(bit1 != bit2 for bit1, bit2 in zip(el1, el2))
    
    # Calculate normalized Hamming distance
    normalized_distance = differing_positions / (3*(len(vector1)))
    
    return normalized_distance
  
def calculateFAR(root_dir, threshold=0.30):
    subdirectories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[:30]

    far_count = 0
    total_comparisons = 0
    HAMMING_THRESHOLD=threshold
    print(threshold)
    # Loop through each subdirectory
    for i, subdir in enumerate(subdirectories):
        subdir_path = os.path.join(root_dir, subdir)
        images = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.bmp', '.jpg', '.jpeg'))])
        
        # Skip subdirectory if no images are found
        if len(images) == 0:
            continue

        # Extract feature vector of the first image in the current subdirectory
        feature_current = process_eye_image(os.path.join(subdir_path, images[0]))

        # Compare the first image of the current subdirectory with the first image of remaining subdirectories
        for j in range(i + 1, len(subdirectories)):
            next_subdir_path = os.path.join(root_dir, subdirectories[j])
            next_images = sorted([f for f in os.listdir(next_subdir_path) if f.endswith(('.bmp', '.jpg', '.jpeg'))])
            
            # Skip if no images are found in the compared subdirectory
            if len(next_images) == 0:
                continue

            # Extract feature vector of the first image in the next subdirectory
            feature_next = process_eye_image(os.path.join(next_subdir_path, next_images[0]))

            # Calculate Hamming distance
            distance = hamming_distance_string(feature_current, feature_next)
            total_comparisons += 1

            # Check if distance is below the threshold (count as False Acceptance)
            if distance < HAMMING_THRESHOLD:
                far_count += 1

    # Calculate FAR
    far = far_count / total_comparisons if total_comparisons > 0 else 0
    print(f"False Acceptance Rate (FAR): {far:.4f}")


def calculateFRR(root_dir, thresh =0.3):
    subdirectories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[:60]

    frr_count = 0
    total_comparisons = 0
    HAMMING_THRESHOLD= thresh
    print(thresh)
    # Loop through each of the top 30 subdirectories
    for subdir in subdirectories:
        subdir_path = os.path.join(root_dir, subdir)
        images = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.bmp', '.jpg', '.jpeg'))])
        
        # Check if at least two images are available in the subdirectory
        if len(images) < 2:
            continue

        # Extract feature vectors for the first two images
        feature1 = process_eye_image(os.path.join(subdir_path, images[0]))
        feature2 = process_eye_image(os.path.join(subdir_path, images[1]))

        # Calculate Hamming distance
        distance = hamming_distance_3String(feature1, feature2)
        total_comparisons += 1

        # Check if distance is below the threshold (count as False Rejection)
        if distance > HAMMING_THRESHOLD:
            frr_count += 1

    # Calculate FRR
    frr = frr_count / total_comparisons if total_comparisons > 0 else 0
    print(f"False Rejection Rate (FRR): {frr:.4f}")


# calculateFAR(root_dir,0.28)
# calculateFRR(root_dir, 0.28)
