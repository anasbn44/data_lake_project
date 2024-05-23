from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import cv2,sys,os
import numpy as np

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



class OCRProcessor:
    def __init__(self):
        # Initialisation de l'OCR
        self.ocr = PaddleOCR(use_angle_cls=True, det_db_thresh = 0.4, det_db_box_thresh = 0.5,det_db_unclip_ratio = 1.4, max_batch_size = 32,
                det_limit_side_len = 1000, det_db_score_mode = "slow", dilation = False,ocr_version='PP-OCRv4', lang='en',use_space_char=True,show_log = False)

    def convert_pdf_to_images(self, pdf_content):
        """
        Convertit un PDF (en contenu de bytes) en une liste d'images PIL.
        """
        print('convert to images....')
        images = []
        try:
            images = convert_from_bytes(pdf_content,dpi=300)
        except MemoryError:
            print(f"Erreur de mémoire: {MemoryError}")
        return images

    def ocr_result(self, image,show=False):
        """
        Resultats OCR.
        """
        if is_background_dark(image):
            image = cv2.bitwise_not(image.copy())  # Inverser l'image
        image = enhance_text_quality(image)
        # if show:
        # # Show the Image in the Window
        #     win_name='Process Image For OCR'
        #     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        #     cv2.imshow(win_name, image.copy())
        #     # Resize the Window
        #     cv2.waitKey(10000)  # Affiche l'image pendant 2000 ms (2 secondes)
        #     cv2.destroyAllWindows()
        #
        result = self.ocr.ocr(image, cls=True)[0]
        return result
    

    def image_to_text(self, image):
        """
        Convertit une image PIL en texte en utilisant OCR.
        """

        result = self.ocr_result(image)
        text=""
        if result:
            text = " ".join([line[1][0] for line in result])
        return text
    

    def process_pdf(self, pdf_content):
        """
        Traite un PDF complet, convertit chaque page en image et extrait le texte.
        Retourne une liste de textes extraits de chaque page.
        """
        images = self.convert_pdf_to_images(pdf_content)
        text_results = [self.image_to_text(image) for image in images]
        return text_results


def is_background_dark(image, threshold=100):
    """ Vérifie si l'image a un fond sombre en analysant les coins de l'image. """
    corners = [
        image[0:50, 0:50],       # Coin supérieur gauche
        image[0:50, -50:],       # Coin supérieur droit
        image[-50:, 0:50],       # Coin inférieur gauche
        image[-50:, -50:]        # Coin inférieur droit
    ]
    corners_brightness = np.mean([np.mean(corner) for corner in corners])
    return corners_brightness < threshold

def enhance_text_quality(image):
    """ Applique un flou gaussien suivi par un seuillage adaptatif pour améliorer la qualité du texte. """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Appliquer un filtre de netteté
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(thresh, -1, sharpen_kernel)
    return sharpened




# Exemple d'utilisation
# ocr_processor = OCRProcessor()
# with open('example.pdf', 'rb') as file:
#     pdf_content = file.read()

# texts = ocr_processor.process_pdf(pdf_content)
# for page_text in texts:
#     print(page_text)
