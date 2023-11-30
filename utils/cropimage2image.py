from PIL import Image
import numpy as np


def cropimage2image(original_image_path, mask_path, background_image_path, x, y, scale):
    original_image = Image.open(original_image_path).convert("RGB")
    mask = Image.open(mask_path)
    background_image = Image.open(background_image_path).convert("RGB")

    original_image_array = np.array(original_image)
    mask_array = np.array(mask)
    background_image_array = np.array(background_image)

    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = original_image_array[min_y:max_y+1, min_x:max_x+1]
    extracted_content_mask=mask_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h)) #(max_x - min_x + 1, max_y - min_y + 1))
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h)) #(max_x - min_x + 1, max_y - min_y + 1))

    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[y:y+scaled_h, x:x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

if __name__ == '__main__':
    x = 10 # box left-uppon 
    y = 50 
    scale = 0.2
    # 读取原始图像、抠出的内容和新的背景图像
    original_image_path = "/home/qifan/datasets/coco/object_sample1/clock/batchV2_1_0.png"
    mask_path = original_image_path.replace("png","_mask.png")
    background_image_path = "/home/qifan/datasets/coco/train2017/000000257946.jpg"
    # 保存路径
    save_path="/home/qifan/Annotation-anything-pipeline/utils/result_image.jpg"
    result_image = cropimage2image(original_image_path, mask_path, background_image_path, x, y, scale)
    result_image.save(save_path)



    
               
                

        
