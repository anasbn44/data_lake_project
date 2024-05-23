import os
import layoutparser as lp
from pdf2image import convert_from_bytes
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import cv2
from PIL import Image
import numpy as np
import io ,sys

import get_data.kpis_search as kpis_search

import warnings

warnings.filterwarnings("ignore")

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

os.makedirs('Graphes_by_pages', exist_ok=True)

# PubLayNet
print('load graph models ....')
modeldetect = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

modeldata = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
print('Done!')

def clean_up_images(folder='Graphes_by_pages'):
    image_extensions = ['.png', '.jpg', '.jpeg']  
    for filename in os.listdir(folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Pour supprimer le fichier
                print(f"Supprimé: {file_path}")
            except Exception as e:
                print(f'Échec de suppression de {file_path}. Raison: {e}')




def main(ocr,images, kpi, search,rapportEco=False,reverse=False):

    os.makedirs('Graphes_by_pages', exist_ok=True)

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

        # Charger l'image depuis le buffer dans OpenCV
        image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convertir de BGR à RGB pour affichage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Créer un objet PIL Image à partir de l'image OpenCV
        pil_img = Image.fromarray(image_rgb)


        win_name='Image Originale'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)   
        # Show the Image in the Window
        cv2.imshow(win_name, image_rgb.copy())
        # Resize the Window
        cv2.resizeWindow(win_name, 800, 1300)
        cv2.waitKey(10000)  # Affiche l'image pendant 2000 ms (2 secondes)
        cv2.destroyAllWindows()
    
        # detect
        print('graph detection ....')
        layout = modeldetect.detect(pil_img)
        print('Done!')

        x_1=0
        y_1=0
        x_2=0
        y_2=0

        im = pil_img.copy()
        im= np.array(im)


        figures = [block for block in layout._blocks if block.type == 'Figure']

        for l in figures:
                x_1 = int(l.block.x_1)
                y_1 = int(l.block.y_1)
                x_2 = int(l.block.x_2)
                y_2 = int(l.block.y_2)

                # Création d'un nom de fichier unique pour chaque figure
                img=im[y_1:y_2, x_1:x_2]
                # Convert cropped image to grayscale
                gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                
                text=ocr.image_to_text(gray)
                print(text)
                if search.lower() in text.lower():
                    image=img
                    win_name='Image Recadree'
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                    # Show the Image in the Window
                    cv2.imshow(win_name, image_rgb)
                    # Resize the Window
                    cv2.resizeWindow(win_name, 800, 1300)
                    cv2.waitKey(10000)  # Affiche l'image pendant 2000 ms (2 secondes)
                    cv2.destroyAllWindows()  
                    

                    if rapportEco== True:
                        
                        scale_width = 0.4 # réduire la largeur de 40%
                        scale_height = 0.5  # réduire la hauteur de 50%

                        # Nouvelles dimensions
                        new_width = int(image.shape[1] * scale_width)
                        new_height = int(image.shape[0] * scale_height)

                        # Redimensionner l'image
                        image = cv2.resize(image, (new_width, new_height))
            
                    print('extracting graph data ....')
                    
                    
                    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
                    predictions = modeldata.generate(**inputs, max_new_tokens=512)
                    print(processor.decode(predictions[0], skip_special_tokens=True))

                    outp=processor.decode(predictions[0], skip_special_tokens=True)
                    clean_up_images()

                    return outp,pagenbr

                else:
                    print(f"graphe {kpi} avec mot-cle <{search}> n'est pas trouvee")
                    return '',''
                    



    else:
        print(f"Le mot {kpi} n'a pas été trouvé dans le document.")
        return '',''
            






