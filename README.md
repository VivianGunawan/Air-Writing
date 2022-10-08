# Air-Writing

🗓 Summer 2018 

## Project  Description
A script to match the trajectory of a colored sticker in a live video feed to an alphanumeric character with 95% accuracy.
Designed a convolutional neural network architecture achieving a 99% test accuracy on the EMNIST dataset within 6 training epochs.

This project was created for AI4All program hosted in Boston University.

## Demonstration

![demo](AirWriting.gif)

## Technical Details

### Creating the Tracker

#### Masking Video Feed by Color

##### Color Representation Choice
The HSV(hue, saturation, value) colorspace designed in the 1970s by computer graphics researchers to more closely align with the way human vision perceives color-making attributes. In these models, colors of each hue are arranged in a radial slice, around a central axis of neutral colors which ranges from black at the bottom to white at the top. The HSV representation models the way paints of different colors mix together, with the saturation dimension resembling various shades of brightly colored paint, and the value dimension resembling the mixture of those paints with varying amounts of black or white paint.

The reason we use HSV colorspace for color detection/thresholding over RGB/BGR is that HSV is more robust towards external lighting changes. This means that in cases of minor changes in external lighting (such as pale shadows,etc. ) Hue values vary relatively lesser than RGB values.

For example, two shades of red colour might have similar Hue values, but widely different RGB values. In real life scenarios such as object tracking based on colour,we need to make sure that our program runs well irrespective of environmental changes as much as possible. So, we prefer HSV colour thresholding over RGB.

##### Cleaning Mask and Finding Contour
Apply erosion and dilation to remove noise in mask.
Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
## Tools
