import os
import os.path as osp
import glob
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree


if __name__ == "__main__":

    anno_mask_path = 'data_cityscape/gt_fine/train'
    anno_mask_list = glob.glob(osp.join(anno_mask_path, '*/*.json'))
    anno_mask_path_test = 'data_cityscape/gt_fine/train/acchen/aachen_000000_000019_gtFine_polygons.json'

    for anno_mask in anno_mask_list:
        # print(anno_mask)
        file_name = anno_mask.split('/')[-1].split('.')[0]
        print(file_name)
        # 이미지에 대해서
        with open(anno_mask) as json_file:
            json_data = json.load(json_file)

            img_height = json_data['imgHeight']     # Image Height
            img_width = json_data['imgWidth']       # Image Widht
            img_objects = json_data['objects']      # Object (polygons)
            objects_num = len(img_objects)          # Object Num

            # Make XML
            root = Element('annotation')
            SubElement(root, 'folder').text = 'cityscape_hazy'
            SubElement(root, 'filename').text = 'test.png'    ## 수정

            size = SubElement(root, 'size')
            SubElement(size, 'width').text = str(img_width)
            SubElement(size, 'height').text = str(img_height)
            SubElement(size, 'depth').text = str(3)

            SubElement(root, 'segmented').text = '0'

            # 물체들에 대해서
            for i, obj in enumerate(img_objects):
                x_min = img_width
                y_min = img_height
                x_max = 0
                y_max = 0
                
                # 물체의 Polygon 들에 대해서
                for inf in obj['polygon']:
                    x = inf[0]
                    y = inf[1]
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # print('min({}, {}), max({}, {}), width:{}, height:{}'.format(x_min, y_min, x_max, y_max, x_max-x_min, y_max-y_min))

                class_usage=True
                # Make Bounding Box
                if obj['label']=='person' or obj['label']=='rider':
                    obj_name = 'person'
                elif obj['label']=='bicycle':
                    obj_name = 'bicycle'
                elif obj['label']=='car' or obj['label']=='truck':
                    obj_name = 'car'
                elif obj['label']=='bus':
                    obj_name = 'bus'
                elif obj['label']=='motorcycle':
                    obj_name = 'motorbike'
                else:
                    class_usage=False
                    
                if class_usage:
                    obj_bnbbox = [x_min, y_min, x_max, y_max]
                    # print('obj_name:{}, obj_bnbbox:{}'.format(obj_name, obj_bnbbox))
                    
                    obj = SubElement(root, 'object')
                    SubElement(obj, 'name').text = obj_name
                    SubElement(obj, 'pose').text = 'Unspecified'
                    SubElement(obj, 'truncated').text = '0'
                    SubElement(obj, 'difficult').text = '0'
                    bbox = SubElement(obj, 'bnbbox')
                    SubElement(bbox, 'xmin').text = str(x_min)
                    SubElement(bbox, 'ymin').text = str(y_min)
                    SubElement(bbox, 'xmax').text = str(x_max)
                    SubElement(bbox, 'ymax').text = str(y_max)
            tree = ElementTree(root)
            tree.write('data_cityscape/test.xml', encoding='utf-8')