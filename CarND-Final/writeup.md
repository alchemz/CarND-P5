**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/not_car.png
[image2]: ./output_images/car.png
[image3]: ./output_images/hog_YCrCb.png
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/sliding_window.png
[image6]: ./output_images/heatmap.png
[image7]: ./output_images/heatmap.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook(Train Classifier)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

I tried various combinations of parameters and trained SVM classifer using only HOG features.
And the best result is using 2 cells per block, and YUV, YCrCb are the best color space to train the classifier.



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To shuffle the train set, I have the image data splited using sklearn.train_test_split, and to normalize the features, I use sklearn.StandardScaler. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
In the Implement Vehicle Tracking and Detection Notebook, I applied the find_car function as the sliding window search. It takes (img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, color_spcae, cells_per_step) as input, and returns car_windows as output.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I combined three features: HOG, color histogram, spatial bin, which provided a nice result.  Here are some example images:

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=Qr329A7jJrw)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  

To obtain a more robust result, a deque is used to record the heatmap of the last 10 frames, I will sum up the frames in a exponential decay manner, then a threhold is applied to the final heatmap, scipy.ndimage.measurements.label() is used to identity the individual blobs in the heatmap.

I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  



### Here is a frame and their corresponding heatmaps:

![alt text][image5]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the difficulty I have come through is to reduce the false postive result. And I tried to sum up the last 10 frames, and average them to get a better heatmap result. It does help improve the tracking, but in some cases, there are still false positive result existed. To solve this issue, I came up with a exponential decay fashion to sum the heatmap, and the result is more promissing.

