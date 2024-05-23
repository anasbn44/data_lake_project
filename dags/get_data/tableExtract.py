import torch

from paddleocr import PaddleOCR
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import io ,sys
from PIL import Image
import get_data.kpis_search as kpis_search


def resource_path(relative_path):
    # Function for Pyinstaller
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        #print(base_path)
    except Exception:
        base_path = os.path.abspath(".")
        #base_path = sys._MEIPASS
    return os.path.join(base_path,relative_path)


os.makedirs('Tables_by_pages', exist_ok=True)
print('importing models ....')

detection_model =torch.hub.load(resource_path('dags/get_data/ultralytics-yolov5'), 'custom', path=resource_path('dags/get_data/best.pt'), source='local')

imgsz = 440


def table_detection(image):
    #     # Create a Named Window
    # win_name='Image Originale'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # # Show the Image in the Window
    # cv2.imshow(win_name, image.copy())
    #
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    # notArray = Image.fromarray(image)
    pred = detection_model(image, size=imgsz)
    
    print(pred)
    pred = pred.xywhn[0]
    result = pred.cpu().numpy()
    
    return result

def crop_image_df(ocr,image, detection_result,table):

    # image= np.array(image)


    width = image.shape[1]
    height = image.shape[0]
    # print(width, height)
    for i, result in enumerate(detection_result):
        class_id = int(result[5])
        score = float(result[4])
        min_x = result[0]
        min_y = result[1]
        w = result[2]
        h = result[3]

        x1 = max(0, int((min_x-w/2-0.02)*width))
        y1 = max(0, int((min_y-h/2-0.02)*height))
        x2 = min(width, int((min_x+w/2+0.02)*width))
        y2 = min(height, int((min_y+h/2+0.02)*height))
        # print(x1, y1, x2, y2)
        crop_image = image[y1:y2, x1:x2]

        # Utilisation de PaddleOCR pour détecter le texte dans l'image rognée
        text_in_image = ocr.image_to_text(crop_image).lower()
        # Vérifier si le mot 'Export' est dans le texte extrait
        if table.lower() in text_in_image:

            # win_name='Image Recadree'
            # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            # # Show the Image in the Window
            # cv2.imshow(win_name, crop_image.copy())
            # # Resize the Window
            #
            # cv2.waitKey(10000)  # Affiche l'image pendant 2000 ms (2 secondes)
            # cv2.destroyAllWindows()


            return crop_image
    




def draw_boxes(image, boxes, texts):
    for box, text in zip(boxes, texts):
        (top_left, top_right, bottom_right, bottom_left) = box
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        # Draw the bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Put the OCR'ed text
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image


def extract_table(ocr,image):

    output = ocr.ocr_result(image,show=True)
    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    #
    # image_with_boxes = draw_boxes(image.copy(), boxes, texts)
    # win_name='Image with Bounding Boxes'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # # Show the Image in the Window
    # cv2.imshow(win_name, image_with_boxes)
    # # Resize the Window
    #
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
   

    # print(texts)
    probabilities = [line[1][1] for line in output]

    image_height, image_width = image.shape[:2]
    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    horiz_lines = np.sort(np.array(horiz_out))
    # print(horiz_lines)
    vert_lines = np.sort(np.array(vert_out))

    out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]
    unordered_boxes = [vert_boxes[i][0] for i in vert_lines]
    ordered_boxes = np.argsort(unordered_boxes)

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        inter_area = abs(max((x_2 - x_1), 0) * max((y_2 - y_1), 0))
        if inter_area == 0:
            return 0

        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

        iou = inter_area / float(box_1_area + box_2_area - inter_area)
        return iou

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            intersection_box = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])

            for b, box in enumerate(boxes):
                the_box = [box[0][0], box[0][1], box[2][0], box[2][1]]
                if iou(intersection_box, the_box) > 0.09:
                    out_array[i][j] = texts[b]

                
    return pd.DataFrame(np.array(out_array))



def process_single_image(ocr,gray,img,table):
    # Détection des tables
    detection_result = table_detection(gray.copy())

    # Recadrage des tables détectées
    crop_image = crop_image_df(ocr,gray, detection_result,table)


        # Vérifier si le fond est sombre


    print('extracting tables ....')
    # from paddleocr import PPStructure,draw_structure_result,save_structure_res

    # table_engine = PPStructure(show_log=True)

    # result = table_engine(img)
    # # Extraction et sauvegarde des tables en CSV et XLSX
    # save_structure_res(result, 'Tables_by_pages',os.path.basename('BMSFL+Mars+2024.jpg').split('.')[0])

    table = extract_table(ocr,crop_image)

    return table    



def main(ocr,images,kpis,tablesearch,reverse=False):
    """
    Convertit une page spécifique d'un fichier PDF en une image et traite cette image.
    :param pdf_path: Chemin vers le fichier PDF.
    :param page_number: Numéro de la page à traiter.
    """

   
    
    tables=[]
    for kpi in kpis:
        print(f'Search for "{kpi}"....')
        image,pagenbr=kpis_search.find_KPI_page(ocr,images,kpi,reverse)

        if pagenbr:
            print(f"Le mot {kpi} trouvé sur la page {pagenbr}.")
            # Créer un buffer de mémoire pour l'image
            buffer = io.BytesIO()

            # Sauvegarder l'image dans le buffer en format JPEG
            image.save(buffer, format='JPEG')

            # Déplacer le curseur au début du buffer
            buffer.seek(0)


            buffer2 = io.BytesIO()
            # Sauvegarder l'image dans le buffer en format JPEG
            image.save(buffer2, format='JPEG')
            # Déplacer le curseur au début du buffer
            buffer2.seek(0)

            # Charger l'image depuis le buffer dans OpenCV
            gray = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

            img = cv2.imdecode(np.frombuffer(buffer2.read(), np.uint8), cv2.IMREAD_COLOR)
        
     

            # Créer un objet PIL Image à partir de l'image OpenCV
            
            
            table=process_single_image(ocr,gray,img,tablesearch)
            tables.append((table, pagenbr))
        
        else:
            print(f"Le mot {kpi} n'a pas été trouvé dans le document.")
            tables.append((pd.DataFrame(), None))

    return tables


def get_tables(ocr, images_dir, kpis, tablesearch, reverse=False):

    tables = []
    image_files = sorted(os.listdir(images_dir), reverse=reverse)

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        for kpi in kpis:
            if find_KPI_in_image(ocr, image, kpi):
                print(f"Le mot '{kpi}' trouvé dans l'image '{image_file}'.")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                table = process_single_image(ocr, gray, image, tablesearch)
                tables.append((table, image_file))


    return tables