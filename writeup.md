# Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window_result.png
[image5]: ./output_images/bboxes_and_heat.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces I grabbed random images from each of the two classes and displayed them to get a feel for what the different color space outputs looks like.

Here is an example using the `YCrCb` color space and HOG parameters of the Y channel:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and after training and scoring some classifiers I chose the standard 8x8 pixels per cell and 2x2 cells per block with 9 orientations and L2 Hysteresis Block normalisations.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM using a feature combination containing spatial RGB values, Color Histogramms from H, Cr and Cb channels and hog features from Y and S channels. (Lines 29 to 51 in second code cell in [Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb))
The classifier training is in the 3rd Code cell in the IPython Notebook

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at 3 scales in the bottom half of the image and came up with this:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using a feature combination containing spatial RGB values, Color Histogramms from H, Cr and Cb channels and hog features from Y and S channels, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4) and here's a [combination with lane detection](./output_images/project_video_lanes.mp4) from the last project.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three examples and their corresponding heatmaps, the output of `scipy.ndimage.measurements.label()` and the resulting bounding boxes:

![alt text][image5]




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I took the suggested approach from the lesson using hog and color features, training a support vector machine and use sliding windows for object detection.

One main difficulty was to select the best color channels for hog and histogram features, so that the detection is reliable but not too slow because of the many features that have to be calculated.

I tried to use PCA to reduce the number of features, but that made the detection step even slower.

An other difficulty was to choose how many sliding windows will be used, because more windows make the pipeline more robust, but also a lot slower.

Using this approach it is not possible to seperate two cars that are very close two each other. They will result in one big bounding box.

