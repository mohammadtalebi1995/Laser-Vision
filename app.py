import argparse
import io
import os
from PIL import Image
import matplotlib.pyplot as plt # for ploting the scanned line
import pandas as pd
import imutils
from numpy import asarray
from numpy import savetxt
import cv2
import numpy as np
import pypylon.pylon as py
from datetime import datetime
from gevent.pywsgi import WSGIServer
import torch
from flask import Flask, render_template, request, redirect, Response
from flask_cors import CORS
from skimage import restoration
from skimage import img_as_ubyte
from selenium import webdriver
import math
import platform
import json
from noise_filters import Filters
app = Flask(__name__)
CORS(app)
import time
from io import BytesIO
first_device = py.TlFactory.GetInstance().CreateFirstDevice()
icam = py.InstantCamera(first_device)
icam.Open()
pixelfor = icam.PixelFormat.Value
icam.PixelFormat = pixelfor

IS_CAMERA_IN_USE = False

def gen():

    global IS_CAMERA_IN_USE

    while True:
        image = icam.GrabOne(4000) ### 4ms time for grabbing image
        image = image.Array
        #image = cv2.resize(image, (0,0), fx=0.8366, fy=1, interpolation=cv2.INTER_LINEAR)### 2048x2048 resolution or INTER_AREA  inter_linear is fastest for and good for downsizing 
        ret, jpeg = cv2.imencode('.jpg', image)    
        frame = jpeg.tobytes()
        IS_CAMERA_IN_USE = False
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n'
               b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
               b'\r\n' + frame + b'\r\n')

    
@app.route('/exposure', methods=['GET', 'POST'])
def exposure():
    ss = icam.ExposureTime.Value
    max = icam.ExposureTime.GetMax()
    if request.method == 'POST':
       # print(request.form.get('text_exposure'))
        r = int(request.form.get('text_exposure'))
        if r<max:
         icam.ExposureTime.SetValue(r)
         ss = icam.ExposureTime.Value
    return render_template('index.html',result = ss, max = max)
#-----------------------------------------------------------------
    
@app.route('/width1', methods=['GET', 'POST'])
def width1():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_width'))
        print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc()
        
    icam.Width.SetValue(r)
    current_w  = icam.Width.GetValue()
    return render_template('index.html',current_w = current_w)
#-----------------------------------------------------------------

#-----------------------------------------------------------------
    
@app.route('/height1', methods=['GET', 'POST'])
def height1():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_height'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.Height.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
#-----------------------------------------------------------------
    
@app.route('/blacklevel', methods=['GET', 'POST'])
def blacklevel():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_blacklevel'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.BlackLevel.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
@app.route('/novin')
def novin():
 global IS_CAMERA_IN_USE
 try:
    
     # while True:
        # if IS_CAMERA_IN_USE == False:
            # IS_CAMERA_IN_USE = True
            # break
            
     icam.Close()
     
     img = py.PylonImage()
     icam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
     t11 = time.time()
     with icam.RetrieveResult(5000) as result:

            # Calling AttachGrabResultBuffer creates another reference to the
            # grab result buffer. This prevents the buffer's reuse for grabbing.
            img.AttachGrabResultBuffer(result)
            ss = torch.rand(1000)
            if platform.system() == 'Windows':
       
                date1 = datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p")
                filename = "Novinilya_"+date1+".jpeg"
                img.Save(py.ImageFileFormat_Jpeg, filename)
            else:
                filename = "Novinilya_"+date1+".png"
                img.Save(py.ImageFileFormat_Png, filename)

            icam.StopGrabbing()
            
     config_dict = {}
     with open("./calib/setup_conf.json" , "r") as config_file:
        config_dict = json.load(config_file)

     ## Capturing n images and saving it in img_list
     img_list = []    
     for i in range(config_dict["number_of_shots"]):
        icam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        with icam.RetrieveResult(5000) as result:
            img.AttachGrabResultBuffer(result)
            img_list.append(result.Array)
            icam.StopGrabbing()
     
     icam.Open()
     img.Release()
     IS_CAMERA_IN_USE = False

     ## Calculate average of n different images
     img_list = np.array(img_list)
     img = np.mean(img_list , axis=0).astype(np.ubyte)
     
     ## Apply neccessary rotation to the image to straighten the laser line
     if config_dict["manual_rotation"]:
        M = cv2.getRotationMatrix2D( (0, 0), config_dict["rotation_angle"] , 1)
        img = cv2.warpAffine(img, M , (img.shape[0], img.shape[1]) )
     
     ## Remove the disortion of the image
     dcm = np.load('./calib/dcm.calib')
     icm = np.load('./calib/icm.calib')
     if config_dict["undisort_camera"]:
         h,  w = img.shape[:2]
         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(icm, dcm, (w,h), 1, (w,h))
         img = cv2.undistort(img, icm, dcm , None , newcameramtx)
     
     ## Cut Top and Bottom of image
     """ CODE FOR CUTING-OUT """
     
     ## SubPixel
     img = cv2.resize( img , dsize = None, fx = config_dict["scale_factor_x"] , fy = config_dict["scale_factor_y"], interpolation = cv2.INTER_LINEAR )
     
     ## Apply Filters
     if config_dict["wavelet_filter"]:
        img = restoration.denoise_wavelet(img , sigma = 16, wavelet = 'dmey' , mode = 'soft' , )
        img = img_as_ubyte(img)
        
     if config_dict["bilat_filter"]:
        img = Filters.bilateral_filter(img , kernel = 5 , sigma_color = 5 , process_log=False)
     
     if config_dict["filter2d_filter"]:
        kernel = np.ones((3,5)) / 30
        kernel[1,2] = 1/2
        img = cv2.filter2D(img , -1 , kernel = kernel)
     
     ## Thresholding
     img_mean = np.mean(img)
     img_std = np.std(img)
     coef = config_dict["threshold_coef"]
     
     ### Apply a cut off based on image standard deviation
     img[ np.where( img <= (img_mean + img_std * coef) ) ] = 0
     
     ### Apply threshold
     thresh = lambda x : cv2.threshold(x , 0 , 255 , cv2.THRESH_OTSU)[1] 
     thresh2 = np.apply_along_axis( thresh , axis = 0 , arr = img ).reshape(img.shape)
        
     ## Mean the image in Y-direction
     indices = np.where(thresh2)
     t1 = time.time()
     X_Ungrided = indices[0]
     Y_Ungrided = indices[1]
     df = pd.DataFrame({'x': X_Ungrided, 'y': Y_Ungrided})
     df = df.groupby('y', as_index=False)['x'].mean()

     
     # X_array = np.round( df[['x']].to_numpy() ).astype(np.int32)
     # Y_array = np.round( df[['y']].to_numpy() ).astype(np.int32)

     # Read the meshgrid that replaces Y for Z
     # mgrid = np.load('mgrid.calib')
     # indices = mgrid[X_array , Y_array]
     # indices = np.squeeze(indices, axis=1)
    
     # X = indices[:,0]
     # Y = indices[:,1]

    ####################################################################################
    
     Y_array = df[['x']].to_numpy()
     X_array = df[['y']].to_numpy()
     t2 = time.time()
     print(f'DF:{t2-t1}')
    ## BackDoorSave
     if config_dict["save_raw"]:
         date1 = datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p")
         filename = "rawdata_"+date1+".numpy"
         with open( filename , "wb" ) as numpyar:
            save = np.stack( [X_array, Y_array] , axis = 1 )
            np.save( numpyar , save )       
    ##
     
     X_array = (config_dict["sensor_pixel_size"] / config_dict["scale_factor_x"]) * (X_array - (config_dict["center_x"] *config_dict["scale_factor_x"]))
     Y_array = (config_dict["sensor_pixel_size"] / config_dict["scale_factor_y"]) * (Y_array - (config_dict["center_y"] *config_dict["scale_factor_y"]) )
     
     Y_array += config_dict["focal_lenght"]
     Y_array **= -1
     Y_array *= -config_dict["focal_lenght"] * config_dict["camera_laser_distance"]
     
     X_array *= Y_array / config_dict["focal_lenght"]
     
     Y_array += config_dict["camera_laser_distance"]
    
     X = X_array
     Y = Y_array
     
     indices = np.stack( [X, Y] , axis = 1 )
     t22 = time.time()
     print(f'Total: {t22-t11}')
     ####################################################################################
     
     ## Calculate the X min/max
     x_min = np.min(X)
     x_max = np.max(X)
     
     ## Calculate the Y min/max
     y_min = np.min(Y)
     y_max = np.max(Y)
     
     ## Return the ratio to 1:1
     difference = max( x_max - x_min , y_max - y_min)
     
     if x_max - x_min > y_max - y_min:
        y_max += (0.5)*difference
        y_min -= (0.5)*difference
            
     else: 
        x_max += (0.5)*difference
        x_min -= (0.5)*difference
        
     data_for_json = list(
        map( lambda x: [ float(x[0]) , float(x[1])] , indices)
     )
    
     final_json = { "data" : data_for_json , "x_min" : float(x_min) , "y_min" : float(y_min) , "x_max" : float(x_max) , "y_max" : float(y_max)}
     final_json = json.dumps(final_json)
     ti = datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p")
     filename_jetson = "Novinilya_"+ti+".json"
     with open(filename_jetson, 'w') as f:
      json.dump(json.JSONDecoder().decode(final_json), f)
            
     return final_json
 
 except Exception as e:
     print(f'\n--------------> ERROR <----------------\n{e}\n##########################################################\n')
     icam.Close()
     return -1
#-----------------------------------------------------------------

@app.route('/NovinGetPlot')
def NovinGetPlot():
  count_grab = 0
  plt.figure(figsize=[12,4])

  try:
     while icam.IsGrabbing():
      count_grab = count_grab + 1
      print(count_grab)
      result = icam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
      if result.GrabSucceeded():
        try:
          # The beef
          Image.fromarray(result.Array).save("1234.png")
        finally:
         result.Release()
  finally:
   #icam.StopGrabbing()
   #icam.Close()
  # img = cv2.imread('123.png',cv2.IMREAD_GRAYSCALE)
   img = cv2.imread('1234.png',cv2.IMREAD_COLOR)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
  #plt.subplot(121); imgplot = plt.imshow(img, cmap="gray")
  (T, thresh2) = cv2.threshold(img, 0 , 255,cv2.THRESH_OTSU)
  img_mask=thresh2[::-1,:]
  indices = np.where(img_mask)
  X = indices[0]
  Y = indices[1]
  
  df = pd.DataFrame({'x': X, 'y': Y})
  #df=df.groupby('y', as_index=False)['x'].mean()

  X_array = df[['x']].to_numpy()
  Y_array = df[['y']].to_numpy()

  indices_X=X_array.transpose()[0]
  indices_Y=Y_array.transpose()[0]

  plt.subplot(122); plt.plot(indices_Y, indices_X, '.')
  plt.savefig('plot.png')

  plt.show()



  # define data
  data = [indices_Y,indices_X]
  # save to csv file
  savetxt('data.txt', data, delimiter=', ')
  dfx = df['x'].to_json()
  dfy = df['y'].to_json()
  data2 = df.to_json(orient='index')
  return data2
#-----------------------------------------------------------------
    
@app.route('/gamma', methods=['GET', 'POST'])
def gamma():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_gamma'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.Gamma.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
#-----------------------------------------------------------------
    
@app.route('/autogain', methods=['GET', 'POST'])
def autogain():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('autogain'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.GainAuto.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------

#-----------------------------------------------------------------
    
@app.route('/autoexposure', methods=['GET', 'POST'])
def autoexposure():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('autoexposure'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.ExposureAuto.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
@app.route('/offsetx', methods=['GET', 'POST'])
def offsetx():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_offsetx'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    #if r <=  icam.OffsetX.GetMax():
    icam.OffsetX.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
#-----------------------------------------------------------------
@app.route('/offsety', methods=['GET', 'POST'])
def offsety():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_offsety'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.OffsetY.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
#-----------------------------------------------------------------
@app.route('/gain', methods=['GET', 'POST'])
def gain():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_gain'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.Gain.SetValue(r)
    index()
    return render_template('index.html')
#-----------------------------------------------------------------
#-----------------------------------------------------------------
@app.route('/digital', methods=['GET', 'POST'])
def digital():
    if request.method == 'POST':
        #print(icam.Width.GetMin())
        r = int(request.form.get('text_digital'))
        #print(r)
       # new_width = icam.Width.GetValue() - icam.Width.GetInc() 
    icam.DigitalShift.SetValue(r)
    return render_template('index.html')
#-----------------------------------------------------------------
@app.route('/')
def index():
    current_w  = icam.Width.GetValue()
    max_w = icam.Width.GetMax()
    min_w = icam.Width.GetMin()
    current_h  = icam.Height.GetValue()

    max_h = icam.Height.GetMax()
    min_h = icam.Height.GetMin()
    
    current_offx  = icam.OffsetX.GetValue()
    max_ox = icam.OffsetX.GetMax()
    min_ox = icam.OffsetX.GetMin()
    
    current_offy  = icam.OffsetY.GetValue()
    max_oy = icam.OffsetY.GetMax()
    min_oy = icam.OffsetY.GetMin()
    current_g  = icam.Gain.GetValue()
    max_g = icam.Gain.GetMax()
    min_g = icam.Gain.GetMin()
    current_b  = icam.BlackLevel.GetValue()
    max_b = icam.BlackLevel.GetMax()
    min_b = icam.BlackLevel.GetMin()
    current_gamma  = icam.Gamma.GetValue()
    max_gamma = round(icam.Gamma.GetMax(), 2)
    min_gamma = icam.Gamma.GetMin()
    current_digital  = icam.DigitalShift.GetValue()
    max_digital = icam.DigitalShift.GetMax()
    min_digital = icam.DigitalShift.GetMin()
    current_exp  = icam.ExposureTime.GetValue()
    max_exposure = icam.ExposureTime.GetMax()
    min_exposure = icam.ExposureTime.GetMin()
    return render_template('index.html',max_w = max_w,min_w = min_w , max_h = max_h, min_h = min_h,
    max_ox = max_ox, min_ox = min_ox , max_oy = max_oy, min_oy = min_oy,
    max_g = max_g,min_g = min_g, min_b = min_b , max_b = max_b,
    max_gamma = max_gamma , min_gamma = min_gamma,max_digital = max_digital, min_digital = min_digital,
    max_exposure = max_exposure,min_exposure = min_exposure , current_w = current_w , current_exp = current_exp
    ,current_h = current_h , current_g = current_g , current_digital = current_digital , current_gamma = current_gamma,
    current_b = current_b , current_offx = current_offx , current_offy = current_offy)

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    '''
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    ).autoshape()  # force_reload = recache latest code
    model.eval()
    '''
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat