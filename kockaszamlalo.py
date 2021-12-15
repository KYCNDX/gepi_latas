import cv2
import numpy as np;

# kep beolvasasa
im = cv2.imread("./kepek/20211215_103149.jpg", 0)

#kep atmeretezese
down_width = 800
down_height = 600
down_points = (down_width, down_height)
resized_down = cv2.resize(im, down_points, interpolation= cv2.INTER_LINEAR)

#threshold beallitas
th, dst = cv2.threshold(resized_down, 127, 255, cv2.THRESH_BINARY);
cv2.imshow("Treshold", dst)
cv2.imwrite("treshold.jpg", dst);

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
params.minCircularity = 0.5

# forma kereksegi szuro
params.filterByConvexity = True
params.minConvexity = 0.57

# elipszis forma szuro
params.filterByInertia = True
params.minInertiaRatio = 0.2

# erzekelo letrehozasa a marameterekkel
detector = cv2.SimpleBlobDetector_create(params)

# pontok kereses
keypoints = detector.detect(dst)

# pontok bejelolese pirossal

im_with_keypoints = cv2.drawKeypoints(dst, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#pottyok megszamlalasa
olvasas_int = len(keypoints)
#szovegge alakitasa
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


# megmutatas
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

#ablak bezarasa
cv2.destroyAllWindows()
