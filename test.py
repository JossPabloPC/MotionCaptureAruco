# Importando las bibliotecas necesarias
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import argparse
import imutils
import time
import cv2
import sys


# Pasando las direcciones como un argumento
DIR = '/content/drive/My Drive/Colab Notebooks/UP_Lectures_202X/TIAv/random_Images/'

args = {
    "image": DIR+"aruco_01.png",
    "type": "DICT_5X5_100",
}
