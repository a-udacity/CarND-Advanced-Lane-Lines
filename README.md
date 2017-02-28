
###Writeup / README
---
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undistorted.png "Undistorted Chessboard"
[image2]: ./output_images/calibration10_undistorted.png "Undistorted Chessboard"
[image3]: ./output_images/calibration11_undistorted.png "Undistorted Chessboard"
[image4]: ./output_images/calibration12_undistorted.png "Undistorted Chessboard"
[image5]: ./output_images/calibration13_undistorted.png "Undistorted Chessboard"
[image5b]: ./output_images/test1_undistorted.png "Undistorted actual image"
[image6]: ./output_images/test1.jpg "Road Transformed"
[image7]: ./examples/binary_combo_example.jpg "Binary Example"
[image8]: ./examples/warped_straight_lines.jpg "Warp Example"
[image9]: ./examples/color_fit_lines.jpg "Fit Visual"
[image10]: ./output_images/straight_lines1_annotated.png "Output"
[image11]: ./output_images/straight_lines2_annotated.png "Output"
[image12]: ./output_images/test1_annotated.png "Output"
[image13]: ./output_images/test2_annotated.png "Output"
[image14]: ./output_images/test3_annotated.png "Output"
[image15]: ./output_images/test4_annotated.png "Output"
[image16]: ./output_images/test5_annotated.png "Output"
[image17]: ./output_images/test6_annotated.png "Output"
[video18]: https://youtu.be/MB3E06JQVwQ "Video"

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. I have provided 5 images of camera calibration side by side.

The code for this step is contained in the third code cell of the IPython notebook located in "./calibrate_camera.ipynb" (Camera calibration class)  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I used the cv2.findChessboardCorners() method to find the imgpoints for each of the calibration images provided. They are appended for each images along with obj points.

Once we have these values (pickled for later use) we can use cv2.calibrateCamera() to get the camera matrix and distortion coefficients. 
Having those we can now undistort a new image using the undistort method in the class that calls cv2.undistort().
I then created a helper function called CameraCalibration->plot_images() to apply the distortion correction to the test images using the `cv2.undistort()` function. Below are the chessboard images undistorted:
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image5b]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a threshold binary image.  Provide an example of a binary image result.
I used yellow threshold (on HSV), Sobel gradient thresholds, and a 2D filter (using cv2.filter2D) to generate a binary image (threshold's steps at function lane_mask in processing.py). Here's an example of my output for this step:

![alt text][image7]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The Perspective class inside perspective.py takes care of this. My source and destination points are defined in detected_lanes.py as shown below:

```
_annotate = dict(
    offset=250,
    src=np.float32([(132, 703), (540, 466), (740, 466), (1147, 703)])
)


detected_lanes_config = dict(
    offset=_annotate['offset'],
    src=_annotate['src'],
    history=7,
    dst=np.float32([(_annotate['src'][0][0] + _annotate['offset'], 720),
                    (_annotate['src'][0][0] + _annotate['offset'], 0),
                    (_annotate['src'][-1][0] - _annotate['offset'], 0),
                    (_annotate['src'][-1][0] - _annotate['offset'], 720)]),
    annotated_video_suffix='_annotated.mp4'
)

```

TODO: This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


![alt text][image8]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Polynomials for left and right lanes are fit on a history of x and y points in the update function in the Line class in lane_detection.py (lines 24-45).

![alt text][image9]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the curvature in the curvature function (lines 124-129) in processing.py.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # 105 through # 129 in my code in `lane_detection.py` in the function `_draw_info` and `_draw_overlay` of class LaneDetector. 
Here are example of my result on a all the test images:


![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/MB3E06JQVwQ)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Getting the pipeline to be of robust against shadows and at the same time capable of detecting yellow lane lines on white ground was difficult. 
I took the approach that lines should never have very low saturation values, i.e. Black. 
Setting a minimal value for the saturation helped when paired with the x gradient and absolute gradient threshold. 
Detecting two lanes was also problematic but using a threshold distance solved the problem.
I struggled the most in implementing the continuous appending of detected lines and then restarting when the buffer of previous lines emptied.

