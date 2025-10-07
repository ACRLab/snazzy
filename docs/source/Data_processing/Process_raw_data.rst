Process raw images
==================

Since the imaging is done with a large Field of View microscope, usually during 6 hours or more, the raw images tend to be in the range of 50 ~ 200 GiB.
The simplest way to handle the raw data is to crop it in individual movies.
There is a considerable amount of background pixels that can be ignored in the raw data. After cropping the embryos, all individual movies combined take about 40% of the original memory space.
This already saves considerable ROM memory but most importantly, it means we can easily load individual movies in the RAM of a regular computer (16-32 GB RAM), without needing to use memory mapped files.

The algorithm to process the raw image can be resumed as:

1. Get the maximum projection of each pixel for the first 10 frames
2. Automatic threshold (Triangle method)
3. Binarize the image 
4. Mark connected regions
5. Select regions based on pixel count
6. Determine the bounding boxes for each movie based on each connected region
7. Open the image (mmap) and save the individual movies as tif

To calculate the bounding boxes of each embryo, we first take the maximum projection of each pixel for the first 10 frames, and then use the Triangle threshold method to binarize the image.
The Triangle threshold is a good choice here because the image has a lot of background pixels, resulting in an unimodal histogram that is centered around the background pixels average value.

Once we have the binary image, we traverse it to identify each embryo.
Whenever a foreground pixel is found, we mark all connecting foreground pixels, and also keep track of the amount of pixels marked and the extreme points (minimum and maximum coordinates in both dimensions).
The pixel count is used to determine if the marked area really corresponds to an embryo, or just a smaller artifact that was erroneously considered a foreground.
The minimum pixel count might change depending on the type of sample being processed, and can be adjusted in ``slice_img.get_bbox_boundaries``.
Regions with high signal intensity, for example corresponding to fly embryo's eyes or gut are examples of smaller artifacts that sometimes are included in the binary image, but can easily be removed due to its size.
The extreme points are then used to generate the bounding boxes, which will determine the positions where the image will be cropped.

The raw image is opened as a memory map using ``numpy``, and the individual embryos are cropped and saved as tif files.
The movies are cropped using a ``ThreadPoolExecutor``.
Because of how Windows machines handle memory mapped file access, it is usually faster to use a single worker on Windows.