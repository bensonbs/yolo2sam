import os
import cv2
import sys
sys.path.append("..")
import json
import torch
import shutil
import zipfile
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from rich.console import Console
from rich.progress import track
from argparse import ArgumentParser
console = Console()

def zip_dir(dir_path, zip_path):
    """
    压缩指定文件夹
    :param dir_path: 目标文件夹路径
    :param zip_path: 压缩文件保存路径+filename.zip
    :return: 无
    """
    zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    for root, dir_names, file_names in os.walk(dir_path):
        # 去掉目标文件夹路径，只对目标文件夹下边的文件及文件夹进行压缩（包括目标文件夹）
        for file_name in file_names:
            zip.write(os.path.join(root, file_name), os.path.join(root.replace(dir_path, ''), file_name))
    zip.close()

def unzip_file(zip_filepath, dest_path):
    """
    解壓縮zip文件
    :param zip_filepath: zip文件路径
    :param dest_path: 解壓后文件存放路径
    """
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def show_mask(mask, image, random_color=False):
    if random_color:
        color = np.random.choice(range(256), size=3)
    else:
        color = np.array([30, 144, 255])
    # mask = cv2.resize(mask,(image.shape[1],image.shape[0])) # 需要根據實際mask的大小和image大小進行調整
    mask = np.repeat(mask[:,:,np.newaxis],3,axis=2)
    mask = mask[:,:,:,0]
    image[mask[:,:,0]!=0] = color

def show_box(box, image):
    start_point = (int(box[0]), int(box[1]))
    end_point = (int(box[2]), int(box[3]))
    color = (0, 255, 0)
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

def get_mask(boxes,image):
    predictor.set_image(image)
    input_boxes = torch.tensor(boxes, device=predictor.device)
    
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    for mask in masks:
        mask = np.moveaxis(mask.cpu().numpy(),0,-1) # 把channel從第一個維度移動到最後一個維度
        show_mask(mask, image, random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), image)
    return masks.cpu(), image

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path-to-zip", help="path to zip",dest="zip_path",default='/home/mefae1/下載/data.zip')
    args = parser.parse_args()
    root, extension = os.path.splitext(args.zip_path)

    console.print("Unzip file..", style="#AAFF00")
    unzip_file(args.zip_path, root)

    json_file = root + '/annotations/instances_default.json'
    labels = json.load(open(json_file, 'r', encoding='utf-8'))

    sam_checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cuda:0"
    console.print("Loading SAM Model..", style="#7DF9FF")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    image_len = len(labels['images'])
    l = 0
    console.print(" :smiley: Start Predict SEG", style="#AAFF00")
    for index in track(range(image_len)):
        image_path = labels['images'][index]['file_name']
        image = cv2.imread(root + '/images/' + image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = []  # to store the boxes
        for annotation in labels['annotations']:
            if annotation['image_id'] == index + 1:
                x, y, w, h = annotation['bbox']
                x1,y1,x2,y2 = x,y,int(x+w),int(y+h)
                boxes.append([x1,y1,x2,y2])
                
        if len(boxes) > 0:
            masks,mask_image = get_mask(boxes,image)
        
            for i, (mask,label) in enumerate(zip(masks,labels['annotations'])):
            
                binary_mask = masks[i].squeeze().numpy().astype(np.uint8)
            
                # Find the contours of the mask
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) ==0 : continue
                largest_contour = max(contours, key=cv2.contourArea)
            
                # Get the new bounding box
                bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
            
                # Get the segmentation mask for object 
                segmentation = largest_contour.flatten().tolist()
                labels['annotations'][l]['segmentation'] = [segmentation]
                l += 1


    # Serializing json
    json_object = json.dumps(labels, indent=4)
    
    # Writing to sample.json
    with open(json_file, "w") as outfile:
        outfile.write(json_object)

    shutil.rmtree(os.path.join(root,'images'))
    zip_dir(root,root+'-seg.zip')
    shutil.rmtree(os.path.join(root))

    console.print("Finshed, output in " + root+ "-seg.zip", style="#AAFF00")