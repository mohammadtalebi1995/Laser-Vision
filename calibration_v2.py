import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
## Considering that camera is at position (0,0)

FOCAL_LENGTH = 16.0 #mm
SENSOR_PIXEL_SIZE = 3.45e-3
CAMERA_LASER_DISTANCE = 174
CENTER_Y = 1024
CENTER_X = 1224

IMAGE_SIZE = (2448 , 2048)

grid = np.mgrid[0:IMAGE_SIZE[0] , 0:IMAGE_SIZE[1]].T.reshape(IMAGE_SIZE[0]*IMAGE_SIZE[1],2)
grid = grid.astype(np.float32)
grid[:,1] = SENSOR_PIXEL_SIZE * (grid[:,1] - CENTER_Y)
grid[:,0] = SENSOR_PIXEL_SIZE * (grid[:,0] - CENTER_X)

grid[:,1] += FOCAL_LENGTH
grid[:,1] **= -1
grid[:,1] *= -FOCAL_LENGTH * CAMERA_LASER_DISTANCE

grid[:,0] *= grid[:,1] / FOCAL_LENGTH

mgrid_trans = grid.reshape(IMAGE_SIZE[1],IMAGE_SIZE[0],2)

with open( 'mgrid.calib' , 'wb') as file:
    np.save(file , mgrid_trans)
 