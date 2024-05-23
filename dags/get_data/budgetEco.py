# pdf_scraper.py
import asyncio
from playwright.async_api import async_playwright
import get_data.ocr as ocr
from datetime import datetime
import requests
import pandas as pd

async def find_pdf_url(base_url, title):
    pdf_url = None
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        current_page_number = 1
        last_page_number = None

        while True:
            if current_page_number == 1:
                current_url = f"{base_url}/downloads/?tag=Conjoncture+et+pr%C3%A9vision+%C3%A9conomique"
            else:
                page_offset = (current_page_number - 1) * 20
                current_url = f"{base_url}/downloads/?tag=Conjoncture+et+pr%C3%A9vision+%C3%A9conomique&p={page_offset}"

            await page.goto(current_url, wait_until='networkidle')

            link_elements = await page.query_selector_all(f".titre_fichier >> text='{title}'")
            for link_element in link_elements:
                download_link = await link_element.get_attribute('href')
                if download_link:
                    pdf_url = f"{base_url}{download_link}"
                    print(f"Document trouvé : {pdf_url}")
                    break

            if pdf_url is not None:
                break

            if not last_page_number:
                pager_links = await page.query_selector_all(".pager a")
                if pager_links:
                    last_page_link = pager_links[-1]
                    last_page_href = await last_page_link.get_attribute('href')
                    last_page_number = int(last_page_href.split('=')[-1]) // 20 + 1
                else:
                    last_page_number = current_page_number

            if current_page_number >= last_page_number:
                print("Fin de la pagination. Document non trouvé.")
                break
            current_page_number += 1

        await browser.close()
    return pdf_url


async def get_pdf_content():
    year = datetime.now().year
    base_url = "https://www.hcp.ma"
    title = f"Budget économique prévisionnel {year} : La situation économique en {year-1} et ses perspectives en {year} (version français)"

    pdf_url = await find_pdf_url(base_url, title)
    if pdf_url:
        response = requests.get(pdf_url)
        if response.ok:
            return response.content
        else:
            print('Failed to download PDF.')
    return None


def extract_table(pdf_content, kpi, search):
    import get_data.tableExtract as tableExtract
    ocr_processor = ocr.OCRProcessor()
    tables = tableExtract.main(ocr_processor, pdf_content, kpi, search, reverse=True)
    for table, pagenbr in tables:
        if not table.empty:
            return table


def extract_text(pdf_content, keyword):
    import get_data.texteExtract as texteExtract
    ocr_processor = ocr.OCRProcessor()
    text_analyser = texteExtract.TextAnalyser()
    results, pagenbr = text_analyser.extract_data(ocr_processor, pdf_content, keyword, keyword, reverse=True)
    if results != "Keyword not found.":
        df = pd.DataFrame(results)
        df.index = [keyword] * len(df)
        return df