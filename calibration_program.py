import numpy as np
import os
from os import path
import cv2 as cv
import argparse

VERSION = 0.1

INTRINSIC_CAMERA_MATRIX_NAME = 'icm.calib'
DISORTION_CAMERA_MATRIX_NAME= 'dcm.calib'
MEASURMENT_CALIBRATION_MATRIX_NAME = 'mcm.calib'
IMAGE_MESH_GRID_NAME = 'mgrid.calib'

SENSOR_SIZE = 3.45e-3 #mm
IMAGE_SIZE = (2448, 2048)
DEFAULT_FOCAL_LENGTH = 8 #mm

IMAGE_EXTENSION = [
    '.jpg' , '.jpeg' , '.png' , '.bmp' 
]

class Util(object):

    @staticmethod
    def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv.resize(image, dim, interpolation=inter)

class Validation(object):

    @staticmethod
    def validateDirectory(pth):
        state = path.isdir(path)

class Calibration(object):

    def __init__(self , input_dir , output_dir , cb_size = (7,7) , cell_length = 1 , flags = None):
        """_summary_

        Args:
            input_dir (str): path to read calibration images.
            output_dir (str): path to save calibration matrices.
            cb_size (tuple, optional): chessboard size; default to (7,7).
            cell_length (int, optional): length of chessboard cell. Defaults to 1.
            flags (int, optional): OpenCV flags for calibration. 
        """        

        self.__input_dir = input_dir
        self.__output_dir = output_dir
        self.__cb_size = cb_size
        self.__cell_length = cell_length
        self.__flags = flags

        self.__end_criteria = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 0.0001 )  


    def __read_calibration_images(self):

        image_list = []
        file_list = os.listdir(self.__input_dir)

        for fname in file_list:
            if str.lower( path.splitext(fname)[-1] )in IMAGE_EXTENSION:
                image_list.append( path.join(self.__input_dir , fname) )

        return image_list

    def __calibrate(self):

        img_list = self.__read_calibration_images()

        objp = np.zeros( (self.__cb_size[0] * self.__cb_size[1] , 3) , np.float32)
        mgrid = np.mgrid[ 0 : self.__cb_size[0] * self.__cell_length :  self.__cell_length , 0 : self.__cb_size[1] * self.__cell_length : self.__cell_length]
        mgrid = mgrid.T.reshape(-1,2)
        objp[:,:2] = mgrid

        objpoints = [] 
        imgpoints = []

        for fname in img_list:
            image = cv.imread( fname, cv.IMREAD_GRAYSCALE )
            ret, corners = cv.findChessboardCorners(image, self.__cb_size, None)
            # ret, imgp = cv.findCirclesGrid( image , self.__cb_size , flags = cv.CALIB_CB_SYMMETRIC_GRID)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(image, corners, (11,11), (-1,-1), self.__end_criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                cv.drawChessboardCorners( image, self.__cb_size , corners2, ret )
                cv.imshow(f'image', Util.ResizeWithAspectRatio(image , width=800))
                cv.waitKey(50)
        
        
        icm = np.array([[ 8 / SENSOR_SIZE  , 0   , IMAGE_SIZE[0]/2],
                        [ 0  , 8 / SENSOR_SIZE   , IMAGE_SIZE[1]/2],
                        [ 0  , 0   , 1   ]] , dtype=np.float64)

        ret , cm , dc , rv , tv = cv.calibrateCamera(
            objpoints, imgpoints, IMAGE_SIZE, cameraMatrix = icm, distCoeffs = None, 
            flags = cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_FIX_PRINCIPAL_POINT
        )

        tv = np.array(tv)
        estimated_height = tv[:,2].mean() 
        
        mtx = np.array([
            [estimated_height / cm[0,0] , 0 , SENSOR_SIZE * -cm[0,2]],
            [0, estimated_height / cm[1,1] ,  SENSOR_SIZE * -cm[1,2]],
            [0, 0, 1]
        ])

        mgrid = np.mgrid[0:IMAGE_SIZE[0] , 0:IMAGE_SIZE[1]].T.reshape(IMAGE_SIZE[0]*IMAGE_SIZE[1],2)
        one_ar = np.ones((IMAGE_SIZE[0]*IMAGE_SIZE[1], 1))
        mgrid = np.concatenate([mgrid , one_ar],axis = 1)
        mgrid_trans = np.matmul(mgrid , mtx.T).reshape(IMAGE_SIZE[0],IMAGE_SIZE[1],3)

        return mgrid_trans, cm , dc , mtx

    def calibrate(self):
        mgrid , cm , dc , mtx = self.__calibrate()
        matrices_dict = {
            INTRINSIC_CAMERA_MATRIX_NAME : cm,
            DISORTION_CAMERA_MATRIX_NAME : dc,
            MEASURMENT_CALIBRATION_MATRIX_NAME : mtx,
            IMAGE_MESH_GRID_NAME : mgrid
        }
        self.__save_result(matrices_dict)
    
    def __save_result(self , matrices_dict):

        for key , val in matrices_dict.items():
            save_path = path.join(self.__output_dir , key)
            with open(save_path , 'wb') as file:
                np.save(file , val)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Camera Calibration v0.1',
        description= 'Calibrate single camera laser profiler using a chesboard of arbitary size.',
        epilog='Novin Ilia Sanat Co,  Writen by Alireza Khalilian, Nov-2022'
    )

    parser.add_argument('-v' , '--version', action='version', version=f'Laser-Profiling Single Camera Calibration v{VERSION}\n\nNovin Ilia Co. Writen by Alireza Khalilian. Nov-2022')

    parser.add_argument(
        '-i' , '--input' , type=str , required=True, dest='input' , help='Directory of chesboard images.\n'
        )

    parser.add_argument(
        '-s' , '--size' , required=True , dest='size' , type = int ,
        help='Chssboard Size; Number of black-to-black corners inside the chesboard.\n' , nargs=2

    )

    parser.add_argument(
        '-l' , '--length' , required=True , dest='length' , type = float , help='Chessboard cell size in mm. Defaults to 1.\n' , default= 1
    )

    parser.add_argument(
        '-o' , '--out' , type=str , required=False, dest='out' , help='Directory to save the calibration output.\n' , default='.'
    )

    args = parser.parse_args()
    calib = Calibration(args.input , args.out , args.size , args.length )
    calib.calibrate()
