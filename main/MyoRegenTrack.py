"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
Author : Yu Yamaoka(Osaka Univ)
mail:yu-yamaoka@ist.osaka-u.ac.jp)
github : https://github.com/RyuAmakaze/MyoRegenTrack
memo : classifier to myoblast(blue), myofiber(red), large myotube(orange), small myotube(yellow), NONE
"""
import os
import torch
import cv2
from glob import glob
from tqdm import tqdm

#Merge
from natsort import natsorted
import numpy as np

#Inference
import cellpose_utils as utils
from inference_pipeline import load_models, load_models_classifier, feature_extractor, linear_regression, classifier

#for args
import argparse

def ExtractImageEdges(image, resize_factor=0.2):
    """
    Args:
        image [h, w, 3]
        resize_factor (float): Factor by which to resize the image for faster processing.
                               Default is 0.5 (i.e., resize to 50% of original size).

    Returns:
        original_mask : np.array of shape [h, w].
    """

    # Resize the image for faster processing
    new_size = (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))
    resized_image = cv2.resize(image, new_size)

    mask = np.zeros(resized_image.shape[:2], np.uint8)  # generate resized init mask

    # 背景モデルと前景モデルの作成
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (50, 50, resized_image.shape[1] - 100, resized_image.shape[0] - 100)  # get position (x, y, width, height)
    cv2.grabCut(resized_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)  # use GrabCut method

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # change mask

    # Resize the mask back to the original size
    original_mask = cv2.resize(mask2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return original_mask

def pad_to_shape(mask, coloring_mask):
    """
    Args:
        mask : [h1, w1]
        coloring_mask : [h2, w2, 3]
    
    Retrun
        mask : [h3, w3]
        coloring_mask : [h3, w3, 3]
    """
    mask_h, mask_w = mask.shape
    color_h, color_w, _ = coloring_mask.shape
    min_h = min(mask_h, color_h)
    min_w = min(mask_w, color_w)
    
    return mask[0:min_h, 0:min_w], coloring_mask[0:min_h, 0:min_w]

def Compute_RecoveryScore(img, colors, weight):
    """
    Args:
        img : [h, w, 3(rgb)]
        colors : [class_num, 3], CLASS_COLOR array
        weight : [class_num]
    Return:
        recovery_score (float)
        colors_pix [class_num(int)]
    """
    recovery_score = 0
    colors_pix = np.zeros(len(colors))
    
    for i, color in enumerate(colors):
        colors_pix[i] = np.sum(np.all(img == color, axis=-1))
    
    for i, w in enumerate(weight):
        recovery_score += (colors_pix[i] * w)/np.sum(colors_pix)
    
    return recovery_score, colors_pix
    

def Coloring(args):
    CLASS_COLOR = [args.MB_color,
                   args.MF_color,
                   args.LMT_color,
                   args.SMT_color,
                   args.NONE_color]
    CLASS_NAME = ["MB", "MF", "LMT", "SMT"]
    
    #Model Load
    vit_model, fc_model = load_models_classifier(args.VIT_MODEL_PATH , args.LINEAR_MODEL_PATH, class_num=4)
        
    #Save Folder
    filename, _ = os.path.splitext(os.path.basename(args.INPUT_PATH))
    if(args.OUTPUT_DIR is not None):
        save_folder_path = os.path.join(args.OUTPUT_DIR, "MyoRegenTrack_"+filename)
    else:
        input_dir = os.path.dirname(args.INPUT_PATH)
        save_folder_path = os.path.join(input_dir, "MyoRegenTrack_"+filename)
    
    #make image tmp folder
    save_cut_folder_path = os.path.join(save_folder_path, "cut")
    save_color_folder_path = os.path.join(save_folder_path, "cut_color")
    os.makedirs(save_cut_folder_path, exist_ok=True)
    os.makedirs(save_color_folder_path, exist_ok=True)
    
    #cut image
    assert os.path.exists(args.INPUT_PATH), "Not found input file:" + args.INPUT_PATH
    input_img = cv2.imread(args.INPUT_PATH)
    cut_imgs, _, width_num = utils.cut_img(input_img, args.CUT_SIZE)
    
    #coloring each cut image
    y_pred = np.array([])#for annlysis csv
    for cut_index, img in enumerate(tqdm((cut_imgs), desc=args.INPUT_PATH)):
        cut_image_path = os.path.join(save_cut_folder_path, f"{str(cut_index)}_{filename}.png")
        cv2.imwrite(cut_image_path, img)
        cut_h, cut_w, _ = img.shape
    
        #cellpose
        mask_cell = utils.img_to_cellpose(cut_image_path, diameter=args.CELLPOSE_DIAMETER, chan=args.CELLPOSE_CHANNEL, 
                                    min_size=args.CELLPOSE_MINSIZE, gpu_enabled = args.CELLPOSE_GPU, model_path = args.CELLPOSE_MODEL_PATH)
        mask_min = utils.img_to_cellpose(cut_image_path, diameter=args.CELLPOSE_DIAMETER_MIN, chan=args.CELLPOSE_CHANNEL, 
                                    min_size=args.CELLPOSE_MINSIZE, gpu_enabled = args.CELLPOSE_GPU, model_path = args.CELLPOSE_MODEL_PATH)
        masks_cell, _ = utils.obj_detection(mask_cell, class_id=0)
        masks_min, _ = utils.obj_detection(mask_min, class_id=0)#class_id is not relation
        
        #canvas
        canvas_Layer0 =  torch.ones((args.CUT_SIZE, args.CUT_SIZE, 3), dtype=torch.uint8)*255 #min cell using args.CELLPOSE_DIAMETER_MIN
        canvas_Layer1 =  torch.ones((args.CUT_SIZE, args.CUT_SIZE, 3), dtype=torch.uint8)*255 #cell using args.CELLPOSE_DIAMETER
        canvas = [canvas_Layer1, canvas_Layer0]
        canvas_merge =  torch.ones((args.CUT_SIZE, args.CUT_SIZE, 3), dtype=torch.uint8)*255 #
        
        #coloring each cell
        masks_arrays = [masks_cell, masks_min]
        for layer, masks in enumerate(masks_arrays):       
            if masks is not None:
                crop_imgs, _, _  = utils.img_to_patch(masks, cut_image_path, args.PATCH_SIZE)
                for cell_index, cell in enumerate(masks):#classifier section
                    if(utils.Check_White(crop_imgs[cell_index], white_thresh=args.WHITE_THRESH)):
                        predicted_class=len(CLASS_COLOR)-1
                    else:                   
                        feature = feature_extractor(vit_model, crop_imgs[cell_index])
                        cls_predict = classifier(fc_model, feature)
                        predicted_class = cls_predict.item()
                    y_pred = np.append(y_pred, cls_predict.cpu()) 
                    
                    #coloring to canvas
                    for k in range(cut_h):
                        for l in range(cut_w):
                            if(masks[cell_index][k, l]==1):
                                canvas[layer][k, l, :] = torch.ByteTensor(CLASS_COLOR[predicted_class])
                                
        #layer1+layer0
        canvas_merge = canvas[0]
        for k in range(args.CUT_SIZE):
            for l in range(args.CUT_SIZE):
                if (canvas_merge [k, l] == torch.tensor([255, 255, 255], dtype=torch.uint8)).all():
                    canvas_merge [k, l] = canvas[1][k, l]   
                    
        save_color_cut_path = os.path.join(save_color_folder_path, f"{str(cut_index)}_{filename}.png")
        canvas_merge_np = canvas_merge.numpy()#to numpy for save
        cv2.imwrite(save_color_cut_path, canvas_merge_np)#save coloring picture
    
    #Merge
    d = []
    color_files = glob(os.path.join(save_folder_path,"cut_color" ,"*.png"))
    for i in natsorted(color_files):
        img = cv2.imread(i)    # img is 'JpegImageFile' object
        img = np.asarray(img)  # trans img to ndarray by np.asarray 
        d.append(img)          # append img
    stacked_rows = []
    for i in range(0, len(color_files), width_num):
        row = np.hstack(d[i:i+width_num])
        stacked_rows.append(row)
    img_x = np.vstack(stacked_rows)
    
    #edge process
    if(args.APPLY_EDGE==True):
        edge_mask = ExtractImageEdges(input_img)
        edge_mask, img_x = pad_to_shape(edge_mask, img_x)
        img_x[edge_mask==0] = args.NONE_color
        
    #Annlysis
    cell_num = len(y_pred)
    CLASS_COLOR.pop()#remove color of NONE
    recovery_score, pix_array = Compute_RecoveryScore(img_x, CLASS_COLOR, args.RECOVERY_WEIGHT)

    #save
    cv2.imwrite(os.path.join(save_folder_path, f"MyoRegenTrack_{filename}.png"), img_x)
    output_txt_path = os.path.join(save_folder_path, 'output.txt')
    with open(output_txt_path, 'w') as file:
        file.write("cell_num" + ":" + str(cell_num) + "\n")
        for i, pix in enumerate(pix_array):
            file.write(CLASS_NAME[i] + ":" +str(pix/np.sum(pix_array))+"[%]\n")
        file.write("recovery_score" + ":" +  str(recovery_score)+"\n")
    
    return 0
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')

    #Inference Param
    parser.add_argument('--CELLPOSE_GPU', default=True, type=bool, help='gpu available')
    parser.add_argument('--CELLPOSE_MINSIZE', default=40, type=int, help='cell min size of cellpose')
    parser.add_argument('--CELLPOSE_CHANNEL', default=[0,0], help='cellpose channel')
    parser.add_argument('--CELLPOSE_DIAMETER', default=40, type=int, help='diameter for muscle cell(layer1)')
    parser.add_argument('--CELLPOSE_DIAMETER_MIN', default=5, type=int, help='diameter for satellite cell(layer0)')
    parser.add_argument('--CUT_SIZE', default=256, type=int, help='input size of cellpose')
    parser.add_argument('--PATCH_SIZE', default=64, type=int, help='input size of dino and classifier')
    parser.add_argument('--WHITE_THRESH', default=205, type=int, 
                        help='Not specification as cell area. If unwanted extra staining is detected by Cellpose, lower the threshold')
    parser.add_argument('--APPLY_EDGE', default=True, type=bool, 
                        help='Determine whether to color or apply processing only to the outline of the input image.')    
    
    #coloring
    parser.add_argument('--MB_color', default=[181, 81, 63], help='assign BGR array. color of Myoblast. default=blue')
    parser.add_argument('--MF_color', default=[0, 0, 213], help='assign BGR array. color of Myofiber. default=red')
    parser.add_argument('--SMT_color', default=[65, 196, 228], help='assign BGR array. color of small myotube. default=yellow')
    parser.add_argument('--LMT_color', default=[0, 108, 239], help='assign BGR array. color of large myotube. default=orange')
    parser.add_argument('--NONE_color', default=[255, 255, 255], help='assign BGR array. color of non cell. default=white')
    parser.add_argument('--LAYER_THRESH', default=0.5, type=float, help='IoU thresh of priority between layer0 and 1')
    
    #Analysis
    parser.add_argument('--RECOVERY_WEIGHT', default=[0.230, 1, 0.756, 0.367], 
                        help='myoblast(MB), myofiber(MF), large myotube(LMT), small myotube(SMT)')

    #Path
    parser.add_argument('--CELLPOSE_MODEL_PATH', default="../model/Cellpose_finetuned_HEstain", 
                        help='Path to cellpose model')
    parser.add_argument('--VIT_MODEL_PATH', default="../model/dino_vitbase8_pretrain.pth", 
                        help='Path to feature extarct model')
    parser.add_argument('--LINEAR_MODEL_PATH', default="../model/ClassifierMLP_LLP.tar", 
                        help='Path to classifier model')
    parser.add_argument('--INPUT_FOLDER', default=None, help='Path to Input Image')
    parser.add_argument('--INPUT_PATH', default="../image/test.jpg", help='Path to input image folder')
    parser.add_argument('--OUTPUT_DIR', default=None, help='Path to save logs and checkpoints')
    args = parser.parse_args()
    print(args)
    
    #Run multiple times or run once
    if args.INPUT_FOLDER is not None:
        files = glob(os.path.join(args.INPUT_FOLDER, "*.tif"))
        for file in files:
            args.INPUT_PATH = file
            Coloring(args)
    else:
        Coloring(args)