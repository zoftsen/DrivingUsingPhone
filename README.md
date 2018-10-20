# DrivingUsingPhone

Introduction
Driving using phone is an attempt to create a prototype to detect whether a person is using his phone while driving. The devices used are:
  1. Hardwares:
      - Up squared dev board - https://up-shop.org/28-up-squared
      - Intel Real Sense SR300 depth camera - https://software.intel.com/en-us/realsense/sr300
  2. Softwares: 
      - Ubuntu 18.04.01  LTS - http://releases.ubuntu.com/18.04/
      - Python3 - https://www.python.org/download/releases/3.0/
      - Intel RealSense SDK files - https://github.com/IntelRealSense/librealsense
      - Spyder IDE for editing Pyhton scripts - https://www.spyder-ide.org/
      - imagemagick for batch image processing - https://www.imagemagick.org/script/index.php
      - DNN using example of tensorflow-for-poets - https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/
      - OpenCV 2 used to create windows - https://opencv.org/


Description:
1. Run label_image_snap_cam.py using spyder IDE
2. Get closer to the depth camera and pretend to speak on the phone. Press the space bar to detect if the system able to detect phone usage. A window with the detected image will pop up.
3. Pressing space bar without speaking to phone will not detect
4. As since this is a prototype, and less sample data is fed for training, the will be some lack of accuracy. 
        
