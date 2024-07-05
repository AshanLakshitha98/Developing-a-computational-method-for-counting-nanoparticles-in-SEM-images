
import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300

from tensorflow.keras.models import load_model

model_no=1
modelpath='savedmodel/modelUNET_TEM(Model'+str(model_no)+').h5'
#modelpath='savedmodel/modelUNET_TEM(mine1).h5'

from patchify import patchify, unpatchify

img_no=16
inputpath='Dwn_imgs/TEM_all/Img/'+str(img_no)+'_img.tif'


img = cv2.imread(inputpath, cv2.IMREAD_GRAYSCALE)
model = load_model(modelpath)

all_img_patches = []
large_image=img

step=128

patches_img = patchify(large_image, (step, step), step=step)


for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i,j,:,:]
        single_patch_img = (single_patch_img.astype('float32')) / 255.
        all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)
images = np.expand_dims(images, -1)
X_test=images

predict=patches_img*0
for x in range(0,patches_img.shape[0]):
    for y in range(0,patches_img.shape[1]):
        patch_img = patches_img[x,y,:,:]
        test_img = (patch_img.astype('float32')) / 255.
        test_img2 = test_img.reshape((1, step, step,1))
        test_img_input=test_img2
        output=((model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8));
        predict[x][y]=output
        print(x,y)

size=(step*patches_img.shape[0],step*patches_img.shape[1])
reconstructed_image = unpatchify(predict, size)
plt.imshow(reconstructed_image, cmap='gray')

#reconstructed_image = reconstructed_image * 255
#output_path = 'output/tst9.2.4.1_1(2).jpg'
#cv2.imwrite(output_path, reconstructed_image)

from tqdm import tqdm

plt.figure(figsize=(9, 9))
square = 6
ix = 1
for i in range(5):
    for j in range(6):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot 
        plt.imshow(predict[i, j, :, :])
        ix += 1
plt.show()

#Predict on large image
large_image = cv2.imread(inputpath, 0)
large_image_scaled = large_image /255.
large_image_scaled = np.expand_dims(large_image_scaled, axis=2)

patch_size=128

import numpy as np
import scipy.signal
from tqdm import tqdm

import gc


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
else:
    PLOT_PROGRESS = False


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)      #SREENI: Changed from 3, 3, to 1, 1 
        wind = wind * wind.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(ret)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):            #SREENI: Changed padx to pady (Bug in original code)
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()

    subdivs = pred_func(subdivs)
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()

    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):                #SREENI: Changed padx to pady (Bug in original code)
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero domain. This may require augmenting the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, 1, pred_func)  # nb_classes = 1 for binary classification
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[1])  # nb_classes = 1 for binary classification

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    large_image_scaled,    #Must be of shape (x, y, c) --> NOT of the shape (n, x, y, c)
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    #nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict((img_batch_subdiv))
    )
)

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(132)
plt.title('Prediction without smooth blending')
plt.imshow(reconstructed_image, cmap='gray')
plt.subplot(133)
plt.title('Prediction with smooth blending')
plt.imshow(predictions_smooth, cmap='gray')
plt.show()


reconstructed_image = predictions_smooth * 255
output_path = 'output/output_final/'+str(img_no)+'(Model'+str(model_no)+').jpg'
#output_path = 'output/output_final/'+str(img_no)+'(mine1).jpg'
cv2.imwrite(output_path, reconstructed_image)

image = cv2.imread(output_path)
original = cv2.imread(inputpath)

shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
# convert the mean shift image to grayscale, then apply Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
plt.imshow(thresh, cmap='binary')

kernel = np.ones((3,3), np.uint8)

eroded_image = cv2.erode(thresh, kernel, iterations=1)
#plt.imshow(eroded_image, cmap='binary')

from scipy import ndimage

thresh=eroded_image
D = ndimage.distance_transform_edt(thresh)
plt.imshow(D, cmap='binary')

from scipy.ndimage import label, maximum_filter
r=25
height, width = D.shape
Z=D

# Create a boolean mask where True corresponds to local maxima
local_maxima_mask = (Z == maximum_filter(Z, footprint=np.ones((r, r))))

# Label the connected components of local maxima
labeled_maxima, num_features = label(local_maxima_mask)

# Find the coordinates of local maxima
local_maxima_coords = np.argwhere(labeled_maxima)

Z_new=np.zeros((height,width))

for coords in local_maxima_coords:
    i, j = coords
    if Z[i, j]!=0:
        #Z_new[i,j]=Z[i, j]
        Z_new[i,j]=1

from skimage.segmentation import watershed

localMax=Z_new
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then apply the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3,3)))[0]
markers = np.resize(markers, (Z_new.shape))
labels = watershed(-D, markers, mask=thresh)
#print("{} particles found".format(len(np.unique(labels)) - 1))
#plt.imshow(labels,'jet')

import imutils

# loop over the unique labels returned by the Watershed
# algorithm
radius_result=[]
a=[] 
b=[] 
theta=[]
C=[]
len_c=[]
count=0
fig, ax = plt.subplots()
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background' so simply ignore it
    if label == 0:
        continue
    # otherwise, allocate memory for the label region and draw it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    C.append(c)
    len_c.append(len(c))
    
avg_c=np.mean(len_c)
std_c=np.std(len_c)
for c in C:
    if len(c) < avg_c-2*std_c or len(c)<6:
        continue
    ellipse = cv2.fitEllipse(c)
    #a.append(ellipse[1][0]/2)
    #b.append(ellipse[1][1]/2)
    #theta.append(ellipse[2])
    cv2.ellipse(original,ellipse,(0,255,0),2)
    
    # draw a circle enclosing the object
    #((x, y), r) = cv2.minEnclosingCircle(c)
    #radius_result.append(r)
    #circle = patches.Circle((int(x), int(y)), int(r), edgecolor=(0, 0, 1), facecolor='none', linewidth=1)
    #ax.add_patch(circle)
    #ax.add_patch(ellipse)
    #cv2.circle(original, (int(x), int(y)), int(r), (0, 255, 0), 2)
    #cv2.putText(original, "{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    count=count+1
plt.imshow(original,'gray')
#plt.axis('off')


print("{} particles found".format(count))

output_path = output_path[:-4]+'('+str(count)+').jpg'
cv2.imwrite(output_path, original)

