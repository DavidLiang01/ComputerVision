README

************Part 1************
A)
Created a function called downsampling that reduce the size of 
a given image by a factor n.

When the the downsample factor is 16, it upsample the image 10 times 
using the following interpolations:
-nearest neighbor interpolation
-bilinear interpolation
-bicubic interpolation


************Part 2************
A)
Created a function called diagonal_shift_img that shifts the image 
to top right using a kernel size of 200*200 and border type that is black.

B)
Created a function called gaussian_filter that takes a kernel size and 
sigma value as parameters. It creates a gaussian kernel filter.

Created a function called gaussian_img that applies the gaussian filter to
an image.


NOTE: Idealy, the sigma should be (kernel size)/3

C)
Created a function gauss_diff that computes the difference of gaussian where 
one of the gaussian filter has a sigma value = constant(k) *sigma

************Part 3************
A)
Created 2 functions called sobel_x and sobel_y which apply the sobel filter x
and y on an image.

B)
Created a function called pixel_orientation that computes the gradient orientation
map based on the matrices obtained in sobel_x and sobel_y.

C)
Created a function called pixel_magnitude that computes the gradient magnitude 
map based on the matrices obtained in sobel_x and sobel_y.

D)
Created a function called canny_img that computes the edges of the image using
the canny algorithm of OpenCV.

************Part 4************
A)
Created a function called non_max_supression that computes the thinning of
edges using non max supression. The function uses the matrices of functions 
pixel_orientation and pixel_magnitude

NOTE: The function will take few seconds to output the image because of the for
loop.