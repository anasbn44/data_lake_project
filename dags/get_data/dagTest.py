from rapport_eco import  download_pdf_content,extract_tables_from_pdf,extract_text_from_pdf
import asyncio

import ocr 

#Class OCR
ocr_processor = ocr.OCRProcessor()

pdf_content=asyncio.run(download_pdf_content())

if pdf_content:
    table_keywords = ['Exportations de Biens et Services', 'Depenses ordinaires']
    text_keywords = [
        'demande etrangere adressee au maroc (hors produits de phosphates et derives)',
        'Cours moyen du baril de Brent',
        'Production cerealiere'
    ]
    tables = extract_tables_from_pdf(pdf_content,ocr_processor, table_keywords)
    print('tables',tables)
    texts = extract_text_from_pdf(pdf_content, ocr_processor,text_keywords)
    print('texts',texts)
else:
    print("Échec du téléchargement du PDF.")
