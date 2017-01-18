
#
# Udacity Self Driving Car Nanodegree
#
# Project 3
# Advanced Lane Finding

**Problem**

Our task is to detect the driving lane on roads using computer vision
techniques.  Here we will engineer specific features from the raw images,
extract these features, and use hand-coded algorithms to detect and track
lanes over time.

**Steps Taken**

We began with a survey of the field spanning developments in the last
20 years.  Lane detection is now a common feature in luxury automobiles
and the computer vision techniques are decades old.  We also followed
instruction from the online Udacity course.   We supplemented this discovery
process with the study of US National Highway standards and survey
papers in civil engineering of road systems.

We created a development environment in Python that pulled together
computer vision algorithms (OpenCV), numerical processing (Numpy),
algorithms (SciPy and Python Anaconda 3.5), visualization (MatPlotLib) and
online lab notebooks (Jupyter).  We ran this as a virtual Anaconda environment
on a 2014-generation MacBook Pro running OS/X El Capitan 10.11.6.

We created a Jupyter notebook for experimentation, available as part of 
this distribution.  The code was procedural and crude as we explored algorithms,
hyper parameter settings, and more.  Once we found settings and algorithms
that gave adequate performance, we refactored the code into a lanefinder.py
library and iterated from there.  We evaluated this library on the project and challenge vidoes and published the results on YouTube for sharing.

**Techniques used**

We explored several techniques and settled on the following basic pipeline:

1. Grab an image
2. Apply filters to detect candidate pixels for lanes
3. Fit a curve to the peaks of pixel intensity on the left and right.

Once we had curves that complied with US standards for distance and curvature,
we modified the algorithm a bit:

1. Grab an image
2. Apply filters as before
3. Use the existing lane curves to extract pixels within a window 
that follows the previous lane markers.  Fit a new curve to these.

**Grab a frame**

Here's what it means to grab an image:

1. Use checkerboard images to determine camera distortion parameters.  Do this
once.

2. Chunk videos into frame images, tensors of shape (height, width, 3),
where the third dimension represents the red, green and blue planes
of an image with 8-bit values per pixel.

3. Resize the images to (360, 640, 3) for faster and consistent processing

4. Remove camera distortion by applying a distortion filter

5. Apply adaptive histogram equalization
AWE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) to improve
contrast and accentuate colors of the base image.

6. Create alternative colorspace views of the image in HSV, YUV, and Gray Scale.

**Apply Filters**

Here's the detail behing "apply filters":

1. Create an empty binary mask we'll use to collect candidate pixels for the
lane marker.

2. Use the HSV (hue, saturation, value/brightness) colorspace to detect
yellow pixels, mark them in the mask.

3. Use the YUV to detect white pixels with Y values above a threshold.  Apply an
11x11 convolution to detect pixels that are 50% brighter than their surroundings.
This will eliminate large objects like cars, reflections and sky as seen in
[this paper](https://www.researchgate.net/publication/275963307_Real-Time_Lane_Detection_and_Rear-End_Collision_Warning_System_on_a_Mobile_Computing_Platform).  Mark these
pixels in our mask.

4. Use the Green plane of the RGB image to find edges with a sobel
filter, add these to the mask.

5. Use the Grayscale version of the RGB to find stark edges, add
these to the mask.

6. Use the Grayscale version of the RGB image to eliminate pixels that
fall below the mean brightness.  This is a simple version of shadow elimination,
focusing on brighter elements.

**Fit a Curve**

Here's the detail behind "fit a curve":

1. Create a histogram of the mean number of 1's along a single veritcal
column of pixels, for all columns across the image.  This gives us a 1-dimensional
vector equal to the width of an image.

2. Apply a smoothing filter to eliminate noise and small perturbations.

3. Find the max mean on the left and right of the image.  Extract a window on
either side of the max column such that all columns have intensity means above
0.05 (5% filled).

4. Use the left, right column markers around these peaks as a vertical slice
of the mask.

5. Fit a 2D polynomial to the 1's that fall within each region as our candidate
lane curve.

**Use Existing Lines**

Once we had curves, here's the detail behind "use existing lane curves":

1. Follow the lane curve from the bottom to the top of the image.  Extract
pixels within 60 pixels on either side as candidates for the new line.  The
line will move.

2. Fit a new curve to these pixels.

3. Ensure that the lane has not shifted by 5%, and that the lane curvature
fits within road standards.  If it does not, discard and use the current
lane.

4. Compute a moving average of the last 10 curves as a true lane marker.

**Future improvements**

The [LaSER](http://cvrr.ucsd.edu/publications/2013/SatzodaLASER13.pdf) algorithm
uses far fewer points than pixels to fit lane curves.  We experimened with this
approach to detect lane boundaries.  The paper talks about 85% accuracy, which gave
us significant problems on the challenge videos.  

We've done a fair job of detecting candidate pixels.  We could improve the 
extraction of lanes from these pixels by exploring algorithms that follow 
contiguous regions.  Morphological transformations can fatten or thin lines
using innovative convolutions, as though we've redrawn our candidate lines
with sharp pencils or heavy, dull markers.

We often grab pixels from the background
that fall above or below a curved marker.  This noise complicates the fitting
process introducing spurious curves, even though lane pixels are clearly shown
in our mask.  Choosing the right pixels gets at the heart of the problem
of detecting lanes.

Our hand-engineered models use very simple constraints for detecting spurious
marker, e.g., whether curvature wildly deviates or the base position shifts
significantly between frames.  More advanced techniques follow the civil engineering
guidelines.

**Failure cases**

Dark shadows, deeply curved roads, bright sun reflecting off the dashboard,
temporary blindness and confusing road pavements confound 
our approach.  Heck, they even confound me when I drive.  The challenge video
shows pavement in progress of being repaired, where the edges of paving
and construction are indeed old lane markers.  I have to admit I avoid
those when I drive and try to switch lanes.  I also wear sunglasses (though
our clahe filter does an admirable job of replicating polarized glasses).

The algorithm also fails at high speeds.  The road curvature will change
rapidly and our averaging technique, as well as the hyperparameters for 
boundary conditions, will change.  We'd need a more adaptive approach
that takes speed into consideration, which the highway department does
all the time when designing roads with curves.

**Conclusion**

This really makes me want to build a deep learning model for lane detection,
creating awesome training data through a laborious process, then using image 
augmentation to create infinite training data for a convolutional
neural network.  My conjecture is that
a model with a few hundred thousand features can be found through nonlinear
optimization far more effectively than iterating through computer vision
techniques by hand on a Mac.

@geohotz was right. Do the Deep Mind thing.
