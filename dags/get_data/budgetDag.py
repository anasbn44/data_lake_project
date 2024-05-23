# main.py
import asyncio
from datetime import datetime
from budgetEco import get_pdf_content, extract_table, extract_text


print('Obtention du contenu PDF ...')
pdf_content = asyncio.run(get_pdf_content())
if pdf_content:
    print('Contenu PDF récupéré avec succès.')

    print("_____________________________________TABLES__________________________________________________")
    kpi = ['annexe']
    search = 'agricole'
    table=extract_table(pdf_content, kpi, search)
    print(table)
    print("_____________________________________TEXTE__________________________________________________")
    keyword = 'compte courant'
    text=extract_text(pdf_content, keyword)
    print(text)
else:
    print("Échec du téléchargement du contenu PDF.")


# v2
pdf_content = asyncio.run(get_pdf_content())
with open(save_path, 'wb') as file:
    file.write(pdf_content)

