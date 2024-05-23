import numpy as np
import cv2,io,os,sys

from pdf2image import convert_from_bytes,pdfinfo_from_bytes

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




def find_KPI_page(ocr, pdf_content, kpi, reverse):
    # Obtenir le nombre de pages dans le PDF
    number_of_pages =pdfinfo_from_bytes(pdf_content)['Pages']

    if reverse:
        rang = reversed(range(number_of_pages))
    else:
        rang = range(number_of_pages)

    # Parcourir chaque page une par une
    for page_number in rang:
        print(f"Search in page {page_number + 1}")

        # Convertir la page courante en image
        images = convert_from_bytes(pdf_content, first_page=page_number + 1, last_page=page_number + 1,dpi=300)
        image = images[0]  # Il n'y a qu'une seule image dans cette liste
        returnImage=image.copy()
        # Sauvegarder l'image en mémoire
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)

        # Charger l'image dans OpenCV
        image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Exécuter l'OCR sur l'image courante
        results = ocr.ocr_result(image)
        full_text = [line[1][0] for line in results]
        # Vérifier si le mot est dans le texte extrait
        if kpi.lower() in ' '.join(full_text).lower():
            return returnImage,page_number + 1  # Retourner le numéro de la page (indexé à 1)

    # Retourner None si 'kpi' n'est pas trouvé
    return None,None


def find_KPI(ocr, image, kpi):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Execute OCR on the image
    results = ocr.ocr_result(image)
    full_text = [line[1][0] for line in results]

    # Check if the KPI is in the extracted text
    return kpi.lower() in ' '.join(full_text).lower()