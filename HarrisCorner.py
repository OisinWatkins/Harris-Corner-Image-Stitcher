import cv2
import numpy as np
from numpy import linalg as la
from scipy import ndimage
import matplotlib.pyplot as plt

"""
Implements a Harris Corner detector, all steps have been implemented globally across the entire input image

Functions accepts a string containing the directory to the png of interest
Returns a thresholded, Non-Max Suppressed Harris Response matrix of the same shape as the input image
"""


def harrisCorner(filename):

    # Import image and convert to grey-scale
    img = cv2.imread(filename)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grey = np.float32(grey)

    Ix = ndimage.gaussian_filter(grey, sigma=1.5, order=(0, 1))
    Iy = ndimage.gaussian_filter(grey, sigma=1.5, order=(1, 0))
    Ix2 = np.power(Ix, 2)
    Iy2 = np.power(Iy, 2)

    A = ndimage.gaussian_filter(Ix2, sigma=3, order=0)
    B = ndimage.gaussian_filter(Iy2, sigma=3, order=0)
    C = ndimage.gaussian_filter(np.multiply(Ix, Iy), sigma=2, order=0)

    detM = np.subtract(np.multiply(A, C), np.power(B, 2))
    trM = np.add(np.add(A, C), 0.000001)
    R = np.divide(detM, trM)
    R_th = (R > 0.01*R.max())  # Threshold at 1% the maximum value

    coords = np.array(R_th.nonzero()).T
    candidate_values = np.array([R[c[0], c[1]] for c in coords])
    indices = np.argsort(candidate_values)

    allowed_locations = np.zeros(R.shape)
    allowed_locations[10:-10, 10:-10] = 1

    filtered_coords = []
    for i in indices[::-1]:
        r, c = coords[i]
        if allowed_locations[r, c]:
            filtered_coords.append((r, c))
            allowed_locations[r-10:r+11, c-10:c+11] = 0

    return filtered_coords


def mergeImages(image1, image2, harris1, harris2):

    Img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    Img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    Harris1 = np.array(harris1)
    Harris2 = np.array(harris2)
    rows1, cols1 = Harris1.shape
    rows2, cols2 = Harris2.shape

    img1PatchVectors = np.zeros([rows1, 121])
    img2PatchVectors = np.zeros([rows2, 121])  # initialise image patch matrices -> 11x11 window used, hence 121 columns needed

    for i in range(rows1):
        window = Img1[Harris1[i, 0] - 5:Harris1[i, 0] + 6, Harris1[i, 1] - 5:Harris1[i, 1] + 6]  # 11x11 patch around point of interest
        window = np.reshape(window, (1, 121))
        window = np.divide(np.subtract(window, np.mean(window)), la.norm(window))

        img1PatchVectors[i, :] = window

    for i in range(rows2):
        window = Img2[Harris2[i, 0] - 5:Harris2[i, 0] + 6, Harris2[i, 1] - 5:Harris2[i, 1] + 6]  # 11x11 patch around point of interest
        window = np.reshape(window, (1, 121))
        window = np.divide(np.subtract(window, np.mean(window)), la.norm(window))

        img2PatchVectors[i, :] = window

    img1PatchVectors = img1PatchVectors[~np.all(img1PatchVectors == 0, axis=1)]
    img2PatchVectors = img2PatchVectors[~np.all(img2PatchVectors == 0, axis=1)]  # Remove any row populated only with zeros

    R = np.dot(img1PatchVectors, np.transpose(img2PatchVectors))  # compute response matrix
    R[R > 0.9*R.max()] = 0  # Threshold applied at 0.9*maximum value

    rowsOfInterest, colsOfInterest = np.nonzero(R)  # Extract indices of remaining translation responses
    numPoints = len(rowsOfInterest)  # no. translations recorded

    translations = np.zeros((numPoints, 2))  # stores the dx and dy info of each ranslation
    a = np.zeros((numPoints, 2))  # stores the image one interest point co-ordinates
    b = np.zeros((numPoints, 2))  # stores the corresponding image 2 interest point co-ordinates
    for i in range(numPoints):
        a[i] = Harris1[rowsOfInterest[i], :]
        b[i] = Harris2[colsOfInterest[i], :]
        translations[i] = [a[i, 0] - b[i, 0], a[i, 1] - b[i, 1]]  # compute all 3 in parrallel

    agreementCount = np.zeros((translations.shape[0]))
    for i in range(translations.shape[0]):
        candidate = translations[i, :]
        opposition = translations[~np.all(translations == translations[i, :], axis=1)]
        for j in range(opposition.shape[0]):
            euclidDist = np.power(np.add(np.power(np.subtract(candidate[0], opposition[j, 0]), 2), np.power(np.subtract(candidate[1], opposition[j, 1]), 2)), 0.5)
            if euclidDist <= 1:
                agreementCount[i] = agreementCount[i] + 1

    bestCandidate = np.where(agreementCount == np.amax(agreementCount))
    avgTranslation = translations[bestCandidate, :]

    # The following if-elif... block determines where to place each image w.r.t. the other image, ie: where should
    # image 1 be initialised to and where in relation to image 1 should image 2 be?
 
    if avgTranslation[0, 0, 0] > 0 and avgTranslation[0, 0, 1] > 0:
        output = np.zeros((image2.shape[0] + np.absolute(avgTranslation[0, 0, 0]).astype(int), image2.shape[1] + np.absolute(avgTranslation[0, 0, 1]).astype(int), 3))  # initialise output to the necessary size

        output[0:image1.shape[0], 0:image1.shape[1]] = image1  # image1 initialised to top left of output
        output[output.shape[0]-image2.shape[0]:output.shape[0], output.shape[1]-image2.shape[1]:output.shape[1]] = image2  # image2 to Bottom Right

    elif avgTranslation[0, 0, 0] < 0 and avgTranslation[0, 0, 1] > 0:
        output = np.zeros((image1.shape[0] + np.absolute(avgTranslation[0, 0, 0]).astype(int), image2.shape[1] + np.absolute(avgTranslation[0, 0, 1]).astype(int), 3))  # initialise output to the necessary size

        output[output.shape[0]-image1.shape[0]:output.shape[0], 0:image1.shape[1]] = image1  # image1 initialised to bottom left of output
        output[0:image2.shape[0], output.shape[1]-image2.shape[1]:output.shape[1]] = image2  # image2 to top right

    elif avgTranslation[0, 0, 0] > 0 and avgTranslation[0, 0, 1] < 0:
        output = np.zeros((image2.shape[0] + np.absolute(avgTranslation[0, 0, 0]).astype(int), image1.shape[1] + np.absolute(avgTranslation[0, 0, 1]).astype(int), 3))  # initialise output to the necessary size

        output[0:image1.shape[0], output.shape[1]-image1.shape[1]:output.shape[1]] = image1  # image1 initialised to top right of output
        output[output.shape[0]-image2.shape[0]:output.shape[0], 0:image2.shape[1]] = image2  # image2 to bottom left

    else:
        output = np.zeros((image1.shape[0] + np.absolute(avgTranslation[0, 0, 0]).astype(int), image1.shape[1] + np.absolute(avgTranslation[0, 0, 1]).astype(int), 3))  # initialise output to the necessary size

        output[output.shape[0]-image1.shape[0]:output.shape[0], output.shape[1]-image1.shape[1]:output.shape[1]] = image1  # image1 initialised to bottom right of output
        output[0:image2.shape[0], 0:image2.shape[1]] = image2  # image2 to top left
    output_uint = output.astype(np.uint8)
    return output_uint

archFile1 = './arch1.png'
archFile2 = './arch2.png'
balloonFile1 = './balloon1.png'
balloonFile2 = './balloon2.png'

archImage1 = cv2.imread(archFile1)
archImage2 = cv2.imread(archFile2)

ArchImage1_copy = archImage1.copy()
ArchImage2_copy = archImage2.copy()

archCorners1 = harrisCorner(archFile1)
archCorners2 = harrisCorner(archFile2)
ArchCorners1 = np.array(archCorners1)
ArchCorners2 = np.array(archCorners2)

for i in range(len(archCorners1)):
    ArchImage1_copy[ArchCorners1[i, 0]-3:ArchCorners1[i, 0]+3:, ArchCorners1[i, 1]-3:ArchCorners1[i, 1]+3] = [0, 0, 255]
cv2.imshow('', ArchImage1_copy)
cv2.waitKey(0)

for i in range(len(archCorners2)):
    ArchImage2_copy[ArchCorners2[i, 0]-3:ArchCorners2[i, 0]+3:, ArchCorners2[i, 1]-3:ArchCorners2[i, 1]+3] = [0, 0, 255]
cv2.imshow('', ArchImage2_copy)
cv2.waitKey(0)

archImgFull = mergeImages(archImage1, archImage2, archCorners1, archCorners2)
cv2.imshow('Compiled Image', archImgFull)
cv2.waitKey(0)

########################################################################################################################

balloonImage1 = cv2.imread(balloonFile1)
balloonImage2 = cv2.imread(balloonFile2)

BalloonImage1_copy = balloonImage1.copy()
BalloonImage2_copy = balloonImage2.copy()

balloonCorners1 = harrisCorner(balloonFile1)
balloonCorners2 = harrisCorner(balloonFile2)
BalloonCorners1 = np.array(balloonCorners1)
BalloonCorners2 = np.array(balloonCorners2)

for i in range(len(balloonCorners1)):
    BalloonImage1_copy[BalloonCorners1[i, 0]-3:BalloonCorners1[i, 0]+3:, BalloonCorners1[i, 1]-3:BalloonCorners1[i, 1]+3] = [0, 0, 255]
cv2.imshow('', BalloonImage1_copy)
cv2.waitKey(0)

for i in range(len(balloonCorners2)):
    BalloonImage2_copy[BalloonCorners2[i, 0]-3:BalloonCorners2[i, 0]+3:, BalloonCorners2[i, 1]-3:BalloonCorners2[i, 1]+3] = [0, 0, 255]
cv2.imshow('', BalloonImage2_copy)
cv2.waitKey(0)

balloonImgFull = mergeImages(balloonImage1, balloonImage2, balloonCorners1, balloonCorners2)
cv2.imshow('Compiled Image', balloonImgFull)
cv2.waitKey(0)

"""
#-----------------------------------------------------------------------------------------------------------------------
plt.imshow(archImage1, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

for i in range(archImage2.shape[0]):
    for j in range(archImage2.shape[1]):
        if archCorners2[i, j] != 0:
            archImage2[i-2:i+2, j-2:j+2] = [0, 0, 255]  # Set the pixels surrounding a detected corner to red
cv2.imshow('', archImage2)
cv2.waitKey(0)
#-----------------------------------------------------------------------------------------------------------------------


=> This is the opencv implementation of the Harris Corner dectection algorithm,
   good idea to compare this with our end result
   
dst = cv2.cornerHarris(grey,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
"""
