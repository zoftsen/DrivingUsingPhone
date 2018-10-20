# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import argparse
import sys
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
    
    float_caster = tf.cast(image_reader, tf.float32)
 
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  

  
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/ImageToDetect/itd.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "Mul"
  output_layer = "final_result"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
graph = load_graph(model_file)
labels = load_labels(label_file)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        
        # Show images
        cv2.namedWindow('Press Space Bar to detect phone usage', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Press Space Bar to detect phone usage', images)
        
        key = cv2.waitKey(1)

      #  if key == 32:
        depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(file_name,depth_gray)
        
        
        t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        with tf.Session(graph=graph) as sess:
        # start = time.time()
         results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
         #end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
          #labels = load_labels(label_file)
        #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
        template = "{} (score={:0.5f})"
        for i in top_k:          
            if (results[i] > 0.90):
                print(template.format(labels[i], results[i]))
               # img=cv2.imread(file_name,0)
               # cv2.imshow(labels[i],img)
                #cv2.putText(images, labels[i], 10,10,0,2,255)
              #  cv2.waitKey(5)
               # cv2.destroyWindow(labels[i])
                break
            #break                
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()