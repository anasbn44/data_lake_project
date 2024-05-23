# pdf_scraper.py
import asyncio
from datetime import datetime
from urllib.parse import urljoin
import requests
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError
import ocr

annee_actuelle = datetime.now().year

async def find_pdf_url(base_url):
    pdf_url = None
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        try:
            await page.goto(base_url, timeout=50000)
            links_selector = 'p a'
            await page.wait_for_selector(links_selector, timeout=5000)
        except TimeoutError:
            print(f"La page {base_url} n'a pas chargé correctement ou le sélecteur n'a pas été trouvé.")
            await browser.close()
            return None

        link_elements = await page.query_selector_all(links_selector)
        for link in link_elements:
            link_text = await link.text_content()
            if f"Note de présentation du projet de la Loi de Finances {annee_actuelle}" in link_text:
                relative_url = await link.get_attribute('href')
                pdf_url = urljoin(base_url, relative_url)
                print(f"URL trouvée : {pdf_url}")
                break

        if pdf_url is None:
            print("Document non trouvé.")
        
        await browser.close()
    return pdf_url

async def get_pdf_content():
    annee_actuelle = datetime.now().year
    base_url = 'https://www.finances.gov.ma'
    full_base_url = f'{base_url}/fr/vous-orientez/Pages/plf{annee_actuelle}.aspx'

    pdf_url = await find_pdf_url(full_base_url)
    if pdf_url:
        try:
            print('Récupération du contenu PDF...')
            response = requests.get(pdf_url)
            response.raise_for_status()
            print('Contenu PDF récupéré avec succès.')
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête: {e}")
    return None

def extract_graph_data(pdf_content, kpi, search):
    from graphData import main as extract_data
    ocr_processor = ocr.OCRProcessor()
    data, page = extract_data(ocr_processor, pdf_content, kpi, search)
    return data, page

def save_graph_data(data, kpi_name):
    if data:
        lines = data.strip().split('<0x0A>')
        entries = lines[2:]
        years, values = [], []
        for entry in entries:
            year, value = entry.split('|')
            years.append(year.strip())
            value = value.strip().replace(',', '.')
            values.append(float(value))
        df = pd.DataFrame({'Année': years, kpi_name: values})
        return df
