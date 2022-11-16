# Importando las bibliotecas necesarias
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import imutils
import cv2
import numpy as np
import sys

# Pasando las direcciones como un argumento
DIR = 'C:/Users/pcjos/OneDrive/Documentos/VS/ComputerVision/Sources/'


parametros = {
    "type": "DICT_5X5_100",
    "video": DIR+"aruco_sample.mp4",
    "output": DIR+"output.avi"
}

#Diccionario de Arucos
ARUCO_DICT = {
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}


arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[parametros["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()


import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML


#Lee el video
video = imageio.mimread(parametros["video"], memtest= "1000MB")  #Loading video

vs = cv2.VideoCapture(parametros["video"])
writer = None

# Iterando sobre los cuadros del video
output = []
bakedAnim = np.matrix([[],[],[]])
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
		i = 0
		# Iterando sobre las coordenadas de las esquinas de cada BB
		for (markerCorner, markerID) in zip(corners, ids):
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# Convirtiendo de coordenadas a enteros
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# Dibujando las BB's
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			# Encontrar el centro de cada ArUCo
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
 

			bakedAnim = np.append(bakedAnim, [[cX], [cY], [0]], axis = 1)
			bakedAnim[i,i] = bakedAnim[i,i] - bakedAnim[0,0] 
			i = i + 1


			# Dibujar el ID
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

	# Checar si el video writer es None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(parametros["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# Guardando la trama con los ArUCos detectados
	output.append(frame)
	writer.write(frame)

print(bakedAnim)
# Liberando los apuntadores inicializados
print("[INFO] cleaning up...")
writer.release()
vs.release()
