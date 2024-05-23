from activity_test import  get_pdf_content,extract_Graphes_from_pdf
import asyncio

import ocr 

#Class OCR
ocr_processor = ocr.OCRProcessor()

pdf_content=asyncio.run(get_pdf_content())

if pdf_content:
    keyword="chomage et sous-emploi"
    search="Ensemble"
    graphe = extract_Graphes_from_pdf(pdf_content,ocr_processor, keyword,search)
    print('Graphe',graphe)
    
