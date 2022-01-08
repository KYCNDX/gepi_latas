import cv2
import numpy as np

#kep beolvasasa, eleresi ut es kep modosithato
img = cv2.imread("./kepek/20220108_152049.jpg", -1 )

#beolvasott kep atmeretezese
down_width = 800
down_height = 600
down_points = (down_width, down_height)
resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)

#arnyekok halvanyitasa, eltuntetese
rgb_planes = cv2.split(resized_down)

result_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)

#arnyek nelkuli kep
result = cv2.merge(result_planes) 

#arnyek nelkuli kep szurkearnyalatos atalakitasa
arnyek_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

#pont kereso parametereinek megadasa
params = cv2.SimpleBlobDetector_Params()

#treshold beallitasok
params.minThreshold = 127
params.maxThreshold = 255

# teruleti szuro - minel kisebb meretet ne vegye pontnak
params.filterByArea = True
params.minArea = 150

# kereksegi szuro - minimalis kerekseg, ami pontnak szamit
params.filterByCircularity = True
params.minCircularity = 0.65

# forma kereksegi szuro - mekkora resz hianyozhat a korbol, amit pontnak vesz
params.filterByConvexity = True
params.minConvexity = 0.57

# elipszis forma szuro - mennyire lehet elipszis formaja a kornek
params.filterByInertia = True
params.minInertiaRatio = 0.2

# erzekelo letrehozasa a megadott parameterekkel
detector = cv2.SimpleBlobDetector_create(params)

# pontok kereses az erzekelovel
keypoints = detector.detect(arnyek_gray)

#pontok berajzolasa a szurkearynalatos es eredeti kepre
im_with_keypoints = cv2.drawKeypoints(arnyek_gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
eredeti = cv2.drawKeypoints(resized_down, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#pontok megszamolasa es szovegge alakitasa
olvasas_int = len(keypoints)
olvasas_str = "pont:"+str(olvasas_int)

#a pontokkal megjelolt kep szurkearnyalatossa alakitasa
keypoints_ff = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)

#zavarszures es objektum hatarok keresese
zavar = cv2.blur(keypoints_ff,(7,7))
hatarok = cv2.Canny(zavar,200,255)

#az ujonnan letrejott objektumok kereses
kernal = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(hatarok, kernal, iterations=41)

contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#az objektumok megszamlalasa es szovegge alakitasa
objects = str(len(contours))

text = "kocka:"+str(objects)

#szoveg formajanak, szinenet es meretenek beallitasa a kepre irashoz
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
bottomLeftCornerOfTXT  = (10,450)
fontScale              = 1
fontColor              = (0,255,0)
thickness              = 2
lineType               = 2

#szam kepre irasa
cv2.putText(eredeti,olvasas_str, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

#kockaszam kepre irasa
cv2.putText(eredeti,text, 
    bottomLeftCornerOfTXT, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)


#eredmeny mutatasa
cv2.imshow("Eredmeny", eredeti)

#varkozas billentyu lenyomasra
cv2.waitKey(0)
#ablak bezarasa
cv2.destroyAllWindows()