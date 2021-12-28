import cv2
import numpy as np

#kep beolvasas
img = cv2.imread("./kepek/20211215_103200.jpg", -1 )

#atmeretezes
down_width = 800
down_height = 600
down_points = (down_width, down_height)
resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)


#arnyek eltuntetes
rgb_planes = cv2.split(resized_down)

result_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)

result = cv2.merge(result_planes)

test = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

#pont kereso parameterei
params = cv2.SimpleBlobDetector_Params()

#treshold beallitasok
params.minThreshold = 127
params.maxThreshold = 255


# teruleti szuro
params.filterByArea = True
params.minArea = 150

# kereksegi szuro
params.filterByCircularity = True
params.minCircularity = 0.7

# forma kereksegi szuro
params.filterByConvexity = True
params.minConvexity = 0.57

# elipszis forma szuro
params.filterByInertia = True
params.minInertiaRatio = 0.2

# erzekelo letrehozasa a marameterekkel
detector = cv2.SimpleBlobDetector_create(params)

# pontok kereses
keypoints = detector.detect(test)

im_with_keypoints = cv2.drawKeypoints(test, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

olvasas_int = len(keypoints)
olvasas_str = str(olvasas_int)

#szoveg beallitasa a kepre irashoz
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 2
fontColor              = (0,255,0)
thickness              = 2
lineType               = 2

#szam kepre irasa
cv2.putText(im_with_keypoints,olvasas_str, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

cv2.imshow("result", result)
cv2.imshow("test", test)
cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
