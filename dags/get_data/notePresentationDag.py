# main.py
import asyncio
from datetime import datetime
from notePresentation import get_pdf_content, extract_graph_data, save_graph_data

pdf_content = asyncio.run(get_pdf_content())

if pdf_content:
    print("\n____________________________________PDF SCRAPPING___________________________________________")
    print("_____________________________________GRAPHES__________________________________________________")
    print("******** KPIs:****************************")
    print("######### Evolution du volume global de l’investissement public")
    print("######### Ratio de l'investissement public par rapport au PIB (en%)\n")

    kpi1 = "Evolution des depenses d'investissement du budget general(en MMDH)"
    search1 = "Evolution des depenses d'investissement du budget general(en MMDH)"
    data1, page1 = extract_graph_data(pdf_content, kpi1, search1)
    graphe1=save_graph_data(data1, 'volume global de l’investissement public')
    print(graphe1)

    kpi2 = "Evolution du Ratio des depenses d'investissement du budget general par rapport au PIB(en %)"
    search2 = "Evolution du Ratio des depenses d'investissement du budget general par rapport au PIB(en %)"
    data2, page2 = extract_graph_data(pdf_content, kpi2, search2)
    print(save_graph_data(data2, "Ratio de l'investissement public par rapport au PIB (en%)"))
else:
    print("Le fichier PDF n'a pas été trouvé ou n'a pas pu être téléchargé.")

