import cv2
import numpy as np
import os

imgQ = cv2.imread('vform.jpeg')

per = 25
# minThreshold0 =350
minThreshold =310
# a = 372<= minThreshold <=480


# roi = [[(239, 982), (242, 1009), 'circle', 'morning']]
roi = [[(614, 487), (647, 509), 'box', 'male'],
       [(662, 489), (694, 512), 'box', 'female'],
       [(222, 742), (254, 767), 'box', 'PHP'],
       [(374, 739), (414, 769), 'box', 'java'],
       [(514, 739), (552, 764), 'box', '.Net'],
       [(632, 737), (669, 767), 'box', 'Graphic Design'],
       [(222, 779), (259, 809), 'box', 'joomla'],
       [(374, 782), (419, 807), 'box', 'Android'],
       [(514, 779), (552, 807), 'box', 'SEO'],
       [(632, 782), (664, 807), 'box', 'Web Design'],
       [(222, 827), (257, 857), 'box', 'Wordpress'],
       [(374, 827), (414, 854), 'box', 'Iphone'],
       [(517, 829), (549, 854), 'box', 'Python'],
       [(634, 827), (672, 857), 'box', 'IOT'],
       [(222, 872), (259, 902), 'box', 'magento'],
       [(222, 927), (259, 949), 'box', 'Branch-Ahmedabad']]

orb = cv2.ORB_create()
orb = cv2.ORB_create(nfeatures=1000)


kp1, des1 = orb.detectAndCompute(imgQ, None)

path = 'v_UserForms'
myPicList = os.listdir(path)
print(myPicList)
print('Total Images {}'.format(len(myPicList)))

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)

    kp2, des2 = orb.detectAndCompute(img, None)
    imgKp2 = cv2.drawKeypoints(img, kp2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)

    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per/100))]
    imgMatches = cv2.drawMatches(img, kp2, imgQ, kp1, good, None, flags=2)

    srcPts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

    h, w = imgQ.shape[:2]
    imgScan = cv2.warpPerspective(img, M, (w, h))
    # print(imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []
    print('####### Extracting Data from Form ')

    for x,r in enumerate(roi):

        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        # cv2.circle(imgMask, (239,994), 13, (0, 255, 0), cv2.FILLED)

        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        # imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]



        if r[2] == 'box':
            imgWarpGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE
            totalPixels = cv2.countNonZero(imgThresh)
            print(totalPixels)
            # if totalPixels[0] > minThreshold0:totalPixels = 1
            if totalPixels > minThreshold:totalPixels = 1
            else:totalPixels = 0
            print(f'{r[3]}: {totalPixels}')
            myData.append(totalPixels)


