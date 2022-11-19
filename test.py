# Importando las bibliotecas necesarias
from imutils.video import VideoStream
from PIL import Image as im
import matplotlib.pyplot as plt
import imutils
import cv2
import numpy as np
import sys

# Pasando las direcciones como un argumento
DIR = 'C:/Users/pcjos/OneDrive/Documentos/VS/ComputerVision/Sources/'


parametros = {
    "type": "DICT_APRILTAG_36h11",
    "video": DIR+"AprilTag01.mp4",
    "output": DIR+"output.avi"
}

#Diccionario de Arucos
ARUCO_DICT = {
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[parametros["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()


import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML


#Lee el video
video = imageio.mimread(parametros["video"], memtest= "1000000MB")  #Loading video

vs = cv2.VideoCapture(parametros["video"])
writer = None

# Total de frames en el video
prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
	else cv2.CAP_PROP_FRAME_COUNT
total = int(vs.get(prop))
print("[INFO] {} total frames in video".format(total))


# Iterando sobre los cuadros del video
output = []

RawAnim = np.zeros((total,2))
print(RawAnim.shape)



i = 0

while True:
	# va leyendo cada frame del video
	(grabbed, frame) = vs.read()
 
	#Si no pudo leer le video
	if not grabbed:
		break

	# Resize a 600 pixeles de ancho
	frame = imutils.resize(frame, width=600)

	# Detectar los ArUCo en está trama específica
	(corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
		arucoDict, parameters=arucoParams)

	# Verificando que al menos se haya detectado un ArUCo
	if len(corners) > 0:
		# A lista
		ids = ids.flatten()
		# Iterando sobre las coordenadas de las esquinas de cada BB
		for (markerCorner, markerID) in zip(corners, ids):
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# Convirtiendo de coordenadas a enteros
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# Encontrar el centro de cada ArUCo
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
 
			RawAnim[i,0] = cX
			RawAnim[i,1] = cY

			i = i + 1

#Normaliza imagen
bakedAnimInt8 =  RawAnim
bakedAnimInt8 = (255*(RawAnim - np.min(RawAnim))/np.ptp(RawAnim)).astype(np.uint8) 

import numpy
a = numpy.asarray(RawAnim)
numpy.savetxt("bakedAnim01.csv", a, delimiter=",")

cv2.waitKey(0)
vs.release()