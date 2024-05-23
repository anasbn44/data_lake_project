import asyncio
from datetime import datetime
from urllib.parse import urljoin
import requests
import pandas as pd
from pdf2image import convert_from_bytes
from playwright.async_api import async_playwright, TimeoutError


annee_actuelle = datetime.now().year

async def find_pdf_url(full_base_url,base_url):
    pdf_url = None
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(full_base_url, timeout=50000)
            links_selector = 'p a'
            await page.wait_for_selector(links_selector, timeout=5000)
            link_elements = await page.query_selector_all(links_selector)
            for link in link_elements:
                if "Rapport économique et financie" in await link.text_content():
                    pdf_url = urljoin(base_url, await link.get_attribute('href'))
                    print(f"URL complète trouvée : {pdf_url}")
                    break
        except TimeoutError:
            print(f"La page {full_base_url} n'a pas chargé correctement.")
        finally:
            await browser.close()
        return pdf_url


async def download_pdf_content():
    base_url = 'https://www.finances.gov.ma'
    pdf_url=await find_pdf_url(full_base_url = f'{base_url}/fr/vous-orientez/Pages/plf{int(annee_actuelle)}.aspx',base_url=base_url)
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


def extract_tables_from_pdf(images, ocr_processor,keywords):
    import tableExtract  
    all_data = []
    for keyword in keywords:
        tables = tableExtract.main(ocr_processor, images, [keyword], keyword, reverse=True)
        for table, pagenbr in tables:
            if not table.empty:
                all_data.append((pagenbr, table))
    return all_data




def extract_text_from_pdf(images,ocr_processor, keywords):
    import texteExtract
    text_analyser = texteExtract.TextAnalyser()
    all_data = pd.DataFrame()
    for keyword in keywords:
        results, pagenbr = text_analyser.extract_data(ocr_processor, images, keyword, keyword, reverse=True)
        if results != "Keyword not found.":
            df = pd.DataFrame(results)
            df.index = [keyword] * len(df)
            all_data = pd.concat([all_data, df])
    return all_data



