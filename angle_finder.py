import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# import util
import pypylon.pylon as py

# DATASET_PATH_1 = 'dataset\d1'
# DATASET_PATH_2 = 'dataset\d3'
# DATASET_PATH_3 = 'dataset\d5'
# DATASET_PATH_4 = 'dataset\d6'

# image_list = list(
#     util.imageNameLister(DATASET_PATH_4).values()
# )
first_device = py.TlFactory.GetInstance().CreateFirstDevice()
icam = py.InstantCamera(first_device)
icam.Open()
pixelfor = icam.PixelFormat.Value
icam.PixelFormat = pixelfor
img = py.PylonImage()

captured_points = False
points = list()
        
while True:

    icam.StartGrabbing(py.GrabStrategy_LatestImageOnly)

    with icam.RetrieveResult(5000) as result:
        img.AttachGrabResultBuffer(result)
        image = result.Array        
        img.Release()
        icam.StopGrabbing()

    image = image.astype(np.ubyte)
    image = cv.threshold(image , 0 , 255 , cv.THRESH_OTSU)[1]
    
    image = image.astype(np.float32)
    mean = image.mean(axis = 1)
    mean = mean[np.where(mean > 10)]
    mean = np.stack([mean , np.arange(mean.size)])
   
    m_y = np.mean( mean[0] )
    m_x = np.mean( mean[1] )
    
    # plt.plot(mean[1,:] , mean[0,:] , 'o-')
    # plt.scatter(x = [m_x] , y = [m_y] , c = 'black')
    # plt.grid(True)
    # plt.show()

    var = np.power( mean.T - np.array([m_y , m_x]) , 2)
    var = np.sqrt( var.T[0] + var.T[1] )
    var = np.mean(var)
   
    point_xs = np.where( np.mean(image , axis = 0) > 1.5 )[0]
    point_ys = np.where( np.mean(image , axis = 1) > 2 )[0]

    slope =  (point_ys[0] - point_ys[-1]) / (point_xs[0] - point_xs[-1])
    angle = np.arctan(slope) * 180 / np.pi
    
    print(f'var = {var}    angle = {angle}' , end = '\r')