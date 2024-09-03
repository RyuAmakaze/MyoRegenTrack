"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
Author : Yu Yamaoka(Osaka Univ)
mail : yu-yamaoka@ist.osaka-u.ac.jp)
github : https://github.com/RyuAmakaze/MyoRegenTrack
"""

from cellpose import models, io
import numpy as np
import os
import cv2

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
def img_to_cellpose(img_path, diameter=None, model_type= 'cyto', chan=[0,0], min_size=40, gpu_enabled = False, model_path = ""):
    """
    Input:
        img_path : (string) Image file PATH
    Return:
        mask : [width, height]
    Args:
        model_type : https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py#L19~L20
        chan : https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py#L209
        min_size : https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py#L175
        gpu_enabled : Are u install cuda?
        model_path : FineTuning model Path
    """
    assert os.path.exists(img_path), ("image path is NOT exist")
    img = io.imread(img_path)
    
    # declare model
    if model_path != "":
        assert os.path.exists(model_path), ("model path is NOT exist")
        model = models.CellposeModel(gpu=gpu_enabled, pretrained_model=model_path)
        mask, _, _ = model.eval(img, diameter=diameter, channels=chan, min_size=min_size)
    else:
        model = models.Cellpose(gpu=gpu_enabled, model_type=model_type)
        mask, flows, styles, diams = model.eval(img, diameter=diameter, channels=chan, min_size=min_size)

    return mask

#trasn 2-d mask to [n, 2-d] masks
def obj_detection(mask, class_id:int):
    """
    Args:
        mask : [width, height](ndarray), image data
        class_id : int , class id(ex : 1day -> 1)
    Return:
        mask : [object num(int), width(int), height(int)]
        cls_idxs : [nobject num(int)]
    """
    data = mask
    labels = []
    for label in np.unique(data):
        if label == 0:
            continue
        else:
            labels.append(label)

    if len(labels) == 0:
        return None, None
    else:
        mask = np.zeros((mask.shape)+(len(labels),), dtype=np.uint8)
        for n, label in enumerate(labels):
            mask[:, :, n] = np.uint8(data == label)
        cls_idxs = np.ones([mask.shape[-1]], dtype=np.int32) * class_id
        
        mask = mask.transpose(2, 0 ,1)#[N, width, height]

        return mask, cls_idxs
    
#cut based on center of bbox in each mask-obj  
def img_to_patch(masks, img_path, size=64):
    """
    Input:
        masks : [n(object num), width, height], n is object num.
        img_path : original image path
        size : cut size
    Return:
        crop_imgs :  [n, size, size, color] (numpy.ndarray), maskの重心を中心にimgからCripした正方画像群
        pad_imgs : [n, size, size, color] (numpy.ndarray), 黒パディングしたcrop_imgs
        positions : [n, 4] (numpy.ndarray), 切り取った画像の座標[n, (height_min, height_max, width_min, width_max)]
    """
    CHANNEL = 3
    
    #load image
    img = cv2.imread(img_path)
    width, height, _ = img.shape
    n, w, h = masks.shape
    assert os.path.exists(img_path), "Not found FilePath"
    assert width>size, "Please enter an image whose width is larger than the crop size."
    assert height>size, "Please enter an image whose vertical width is larger than the crop size."
    assert (w==width)and(h==height), "The size of the input mask and the loaded image do not match."
    
    #args for return
    crop_imgs = np.zeros((size, size, CHANNEL)+(len(masks),), dtype=np.uint8)
    pad_imgs = np.zeros((size, size, CHANNEL)+(len(masks),), dtype=np.uint8)
    positions = np.zeros((len(masks), 4), dtype=np.uint32)
    
    #crop each obj
    for i in range(len(masks)):
        #compute center of cell segmentation
        mu = cv2.moments(masks[i], False)
        g_height, g_width = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
        
        #The cutting position should be a square of size with the center of gravity as the center.
        height_min = g_height - size//2
        height_max = g_height + size//2
        width_min = g_width - size//2
        width_max = g_width + size//2
        
        #Exception handling at the wall
        if height_min<0:#Preventing out-of-array access
            height_min = 0
            height_max= (size//2) * 2
        elif height_max>height:#Preventing out-of-array access
            height_min = height - ((size//2)  * 2 + 1)
            height_max = height - 1
    
        #Exception handling at the wall
        if width_min<0:#Preventing out-of-array access
            width_min = 0
            width_max= (size//2) * 2
        elif width_max>width:#Preventing out-of-array access
            width_min = width - ((size//2)  * 2 + 1)
            width_max = width - 1
            
        #Crop img and mask
        crop_img = img[width_min:width_max, height_min:height_max]
        crop_mask = masks[i][width_min:width_max, height_min:height_max] 
            
        #padding img
        pad_img = np.zeros_like(crop_img)
        for ch in range(CHANNEL):  # Process each RGB channel
            pad_img[:,:,ch] = crop_img[:,:,ch] * crop_mask
        
        #Save to return array
        positions[i] = [height_min, height_max, width_min, width_max]
        crop_imgs[:, :, :, i] = crop_img
        pad_imgs[:, :, :, i] = pad_img

    crop_imgs = crop_imgs.transpose(3, 0 , 1, 2)
    pad_imgs = pad_imgs.transpose(3, 0 , 1, 2)

    return crop_imgs, pad_imgs, positions   
    
    
#Find the center of gravity of BBOX of each mask-obj and cut it out 
def mask_to_patch(masks, img_path, mask_path, size=64):
    """
    Input:
        masks : [n(objnum), width, height], n is object num.
        img_path : original image path
        mask_path : original annotation mask image path
        size : cut size
    Return:
        crop_imgs :  [n(objnum), size, size, color](list(numpy_array))
        crop_masks : [n(objnum), size, size, color](list(numpy_array))
        positions : [n, 4] Coordinates of the cropped image[n, (height_min, height_max, width_min, width_max)]
    """
    assert os.path.exists(img_path), "Not found FilePath"
    
    #laod image
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    width, height, _ = img.shape
    assert width>size, "Please enter an image whose width is larger than the crop size."
    assert height>size, "Please enter an image whose vertical width is larger than the crop size."
    
    #args for return
    crop_imgs = np.zeros((size, size, 3)+(len(masks),), dtype=np.uint8)
    crop_masks = np.zeros((size, size, 3)+(len(masks),), dtype=np.uint8)
    positions = np.zeros((len(masks), 4), dtype=np.uint32)
    
    #crop each obj
    for i in range(len(masks)):
        #compute center of cell segmentation
        mu = cv2.moments(masks[i], False)
        g_height, g_width = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
        
        #The cutting position should be a square of size with the center of gravity as the center.
        height_min = g_height - size//2
        height_max = g_height + size//2
        width_min = g_width - size//2
        width_max = g_width + size//2
        
        #Exception handling at the wall
        if height_min<0:#Preventing out-of-array access
            height_min = 0
            height_max= (size//2) * 2
        elif height_max>height:#Preventing out-of-array access
            height_min = height - ((size//2)  * 2 + 1)
            height_max = height - 1
    
        #Exception handling at the wall
        if width_min<0:#Preventing out-of-array access
            width_min = 0
            width_max= (size//2) * 2
        elif width_max>width:#Preventing out-of-array access
            width_min = width - ((size//2)  * 2 + 1)
            width_max = width - 1
        
        #print([height_min, height_max, width_min, width_max])
        positions[i] = [height_min, height_max, width_min, width_max]
        crop_imgs[:, :, :, i] = img[width_min:width_max, height_min:height_max]
        crop_masks[:, :, :, i] = mask[width_min:width_max, height_min:height_max]

    crop_imgs = crop_imgs.transpose(3, 0 , 1, 2)
    crop_masks = crop_masks.transpose(3, 0 , 1, 2)

    return crop_imgs, crop_masks, positions

def Compute_CellArea(onemask):
    """
    Args:
        onemask : [width, height]
    Return:
        Area : int
    """
    #compute white area
    w, h = onemask.shape
    #white_area = cv2.countNonZero(onemask)
    
    #Compute area, len, cir
    contours, _ = cv2.findContours(onemask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    area = cv2.contourArea(contours[0], True)

    return abs(area)

#Calculate the average color value of clip_img to determine whether it is white or not
def Check_White(onemask, white_thresh=220):
    """
    Args:
        onemask : [width, height, 3]
        white_thresh : 白の閾値
    Return:
        bool
    """
    channel_means = np.mean(onemask, axis=(0, 1))# Calculate the average value for each channel
    if all(mean >= white_thresh for mean in channel_means):
        return True
    else:
        return False
    
def Cell_Position_check(onemask):
    """
    Args:
        onemask : [width, height]
    Return:
        position : [xmin, xmax, ymin, ymax]
    """
    rows, cols = onemask.shape
    
    # Initialize a list to store the x and y coordinates.
    x_coords = []
    y_coords = []

    # Examine the mask to get the x and y coordinates where the value is 1
    for i in range(rows):
        for j in range(cols):
            if cell[i, j] == 1:
                x_coords.append(j)
                y_coords.append(i)
    
    return [ min(x_coords), max(x_coords), min(y_coords), max(y_coords)]

#Returns the object [w,h] with the most IOU to True[n,w,h].
def find_best_iou_match(target_mask, source_masks):
    """
    Find the best IoU match for a given inference mask among true masks.

    Parameters:
    - target_mask: [w, h] input.
    - source_masks: [n, w, h]List of Target masks.

    Returns:
    - best_iou: most IoU in input
    """
    
    def calculate_iou(mask1, mask2):
        """
        Calculate Intersection over Union (IoU) between two masks.

        Parameters:
        - mask1, mask2: [w,h] Binary masks to calculate IoU.

        Returns:
        - IoU value.
        """
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    best_iou = 0
    for source_mask in source_masks:
        iou = calculate_iou(target_mask, source_mask)
        best_iou = max(best_iou, iou)
    return best_iou

def cut_img(img, size):
    """
    Args:
        img : [w, h, 3]
        size : int, cut pixel size 
    Return:
        cut_imgs : (n, [w, h, 3]), list[np.array]
        h_num : int, cut num
        w_num : int, cut num
    Text:
        cut w*h image to square image(args.size)
    """
    h, w, c = img.shape
    h_num = round(h/size)#Calculate the number of heights, and truncate if not divisible
    w_num = round(w/size)#Calculate the number of width, and truncate if not divisible
    cut_num = h_num * w_num
    
    cut_imgs = [img[size*x:size*(x+1), size*y:size*(y+1)] for x in range(h_num) for y in range(w_num)]
    
    return cut_imgs, h_num, w_num