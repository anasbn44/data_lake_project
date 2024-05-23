import asyncio
from playwright.async_api import async_playwright
from datetime import datetime
import requests


year = datetime.now().year
base_url = "https://www.hcp.ma"
title = f"Activité, emploi et chômage, résultats annuels {year-1}"



async def download_pdf_with_pagination(base_url, title):
    global pdf_url
    pdf_url = None
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        found = False
        current_page_number = 1
        last_page_number = None

        while not found:
            # Construire l'URL en fonction de la pagination
            if current_page_number == 1:
                current_url = f"{base_url}/downloads/?tag=March%C3%A9+du+travail"
            else:
                page_offset = (current_page_number - 1) * 20
                current_url = f"{base_url}/downloads/?tag=March%C3%A9+du+travail&p={page_offset}"

            await page.goto(current_url, wait_until='networkidle')

            # Rechercher les liens de téléchargement par titre
            link_elements = await page.query_selector_all(f".titre_fichier >> text='{title}'")
            for link_element in link_elements:
                download_link = await link_element.get_attribute('href')
                if download_link:
                    pdf_url = f"{base_url}{download_link}"
                    print(f"Document trouvé : {pdf_url}")
                    found = True
                    break

            if found:
                break

            # Mise à jour de la pagination
            if not last_page_number:
                pager_links = await page.query_selector_all(".pager a")
                last_page_link = pager_links[-1]
                last_page_href = await last_page_link.get_attribute('href')
                last_page_number = int(last_page_href.split('=')[-1]) // 20 + 1

            if current_page_number >= last_page_number:
                print("Fin de la pagination. Document non trouvé.")
                break
            current_page_number += 1

        await browser.close()
        return pdf_url

async def get_pdf_content():

    pdf_url = await download_pdf_with_pagination(base_url=base_url, title=title)
    if pdf_url:
        try:
            print('Récupération du contenu PDF...')
            response = requests.get(pdf_url)  # Augmentation du timeout à 10 secondes
            response.raise_for_status()  # S'assure que la requête s'est bien passée
            print('Contenu PDF récupéré avec succès.')
            return response.content
        except requests.exceptions.HTTPError as errh:
            print(f"Erreur HTTP: {errh}")  # Problèmes comme les erreurs 404, 500, etc.
            return None
        except requests.exceptions.ConnectionError as errc:
            print(f"Erreur de connexion: {errc}")  # Problème de réseau, DNS, refus de connexion
            return None
        except requests.exceptions.Timeout as errt:
            print(f"Timeout: {errt}")  # Le serveur n'a pas répondu dans le temps imparti
            return None
        except requests.exceptions.RequestException as err:
            print(f"Erreur lors de la requête: {err}")  # Autres erreurs liées aux requêtes
            return None


    return None


def extract_Graphes_from_pdf(pdf_content, ocr_processor,keyword,search):
    from get_data.graphData import main as extract_data
    from io import StringIO
    import pandas as pd

    all_data = []
    graphe = []
    outp,pagenbr = extract_data(ocr_processor,pdf_content, keyword,search)
    if outp != '':
        lignes = outp.strip().split('<0x0A>')

        # Extraction des noms de colonnes à partir de la chaîne (en supprimant les espaces inutiles)
        noms_colonnes = lignes[1].split("|")
        noms_colonnes = [nom.strip() for nom in noms_colonnes]

        # Reconstitution de la chaîne sans la ligne des titres pour la création du DataFrame
        donnees_sans_titres = "\n".join(lignes[2:])

        # Utilisation de StringIO pour simuler la lecture d'un fichier
        donnees3 = StringIO(donnees_sans_titres)

        # Création du DataFrame sans fournir les noms de colonnes explicitement
        graphe = pd.read_csv(donnees3, sep="|", names=noms_colonnes)

    if not graphe.empty:
            filename =title
            page_filename = f'{filename}_page_{pagenbr}'
            all_data.append((page_filename, graphe))
            print(all_data)
    return all_data