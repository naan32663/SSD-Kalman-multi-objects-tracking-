
"""
Created on Thu May  7 19:36:14 2020

@author: Anna
"""

from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import os
from matplotlib import pyplot as plt
 

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

from kalman.tracker import Tracker

# Set the image size.
img_height = 300
img_width = 300


# Clear previous models from memory.
K.clear_session() 

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=7,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

weights_path = 'SSD_300x300_7_classes.h5'

model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

path = "./frames/IMG_8501_ano/"
out_path="./output-k/"

images = os.listdir(path)
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background', 'pedestrian','bicycle','car','motorcycle','bus','truck','scooter']

tracker = Tracker(160, 5, 6, 100)

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127)]

for img in images:

    img_name = str(img).split('.')[0]
    img_path = os.path.join(path,img)
    print(img_name)
    
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.
    
    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img) 
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    # Display the image and draw the predicted boxes onto it.    
 
    plt.figure(figsize=(20,12))
    fig = plt.gcf()
    plt.axis('off')
    plt.imshow(orig_images[0])

    current_axis = plt.gca()
    line = ""
    data = img_name + ";"
    
    centers = []
    poses = []
    number = len(y_pred_thresh[0])
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='small', color='white', bbox={'facecolor':color, 'alpha':1.0})
        data = data + str(box[0]) +"," + str(box[1]) + "," + str(xmin) + "," + str(ymin) + "," + str(xmax-xmin) + "," + str(ymax-ymin) +";"
        center = np.array([[(xmin+xmax)//2],[(ymin+ymax)//2]])
        centers.append(center)
        pos = np.array([xmin, ymax])
        poses.append(pos)
#    print("len of center = " + str(len(centers)))
    result = np.asarray(orig_images[0])#, dtype=np.float32)
    if (len(centers) > 0):

        # Track object using Kalman Filter
        tracker.Update(centers, poses)

        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            xmin = tracker.tracks[i].pos[0]
            ymax = tracker.tracks[i].pos[1]
            boxid = tracker.tracks[i].track_id
            label = "ID: {:0>6d}".format(boxid)
            current_axis.text(xmin, ymax, label, size='small', color='white', bbox={'facecolor':color, 'alpha':1.0})

    out_img_path =os.path.join(out_path,img_name+".jpg")
    fig.savefig(out_img_path, bbox_inches='tight', dpi=120, pad_inches=0.0)
    