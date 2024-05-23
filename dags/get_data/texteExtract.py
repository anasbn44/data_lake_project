import spacy
from spacy.matcher import Matcher
import get_data.kpis_search as kpis_search
import cv2,io,os
import numpy as np
from spacy.language import Language



@Language.component("custom_seg")
def custom_seg(doc):
    for token in doc[:-1]:  # Parcourir les tokens du document
        if token.text in {".", "!","..","...", "?"}:
            doc[token.i + 1].is_sent_start = True
        else:
            doc[token.i + 1].is_sent_start = False
    return doc

os.makedirs('Texte_by_pages', exist_ok=True)
class TextAnalyser:
    def __init__(self, model="fr_core_news_lg"):
        self.nlp = spacy.load(model)
        self.nlp.add_pipe("custom_seg", after="tok2vec")
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_matcher()
    
    def _setup_matcher(self):
        pattern_value = [
            {"LIKE_NUM": True},
            {"IS_PUNCT": True, "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
            {"LOWER": {"REGEX": "[%$]|bbl|dollars|millions"}}
        ]
        pattern_year = [
            {"IS_DIGIT": True, "LENGTH": 4}
        ]
        self.matcher.add("VALUE", [pattern_value])
        self.matcher.add("YEAR", [pattern_year])
    
    def extract_data(self,ocr, images,search, keyword,reverse):
        print(f'Search for "{keyword}"....')
        image,pagenbr=kpis_search.find_KPI_page(ocr,images,search,reverse)
        trimmed_sent = ""
        if pagenbr:
            print(f"Le mot {search} trouvé sur la page {pagenbr}.")
            buffer = io.BytesIO()
            # Sauvegarder l'image dans le buffer en format JPEG
            image.save(buffer, format='JPEG')

            # Déplacer le curseur au début du buffer
            buffer.seek(0)
    

            # Charger l'image depuis le buffer dans OpenCV
            gray = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            # Sauvegarder l'image dans le buffer en format JPEG
            page_text=ocr.image_to_text(gray)
            
   
            doc = self.nlp(page_text)

            for sent in doc.sents:             
                keyword_index = sent.text.lower().find(keyword.lower())  
                if keyword_index != -1:
                    # Trouver le début de la phrase coupée
                    words_before_keyword = 3
                    # Convertir le texte de la phrase en liste de mots pour manipuler les indices
                    words = sent.text.split()
                    keyword_first_word_index =sent.text.lower().split().index(keyword.lower().split()[0])
                    start_index = max(0, keyword_first_word_index - words_before_keyword)
                    trimmed_sent = " ".join(words[start_index:])
                    print(f'\n La phrase trouvee qui discute {keyword} est:\n{trimmed_sent}\n' )
                    
                    break
        
        if trimmed_sent == "":
            return "Keyword not found.",pagenbr
        
        doc = self.nlp(trimmed_sent)
        matches = self.matcher(doc)
        values = []
        years = []
        
        for match_id, start, end in matches:
            span = doc[start:end]
            rule_id = self.nlp.vocab.strings[match_id]
            if rule_id == "VALUE":
                values.append((span, start, end))
            elif rule_id == "YEAR":
                years.append((span, start, end))
        
        results = self._associate_values_with_years(values, years, doc)
        return results,pagenbr

    def _associate_values_with_years(self, values, years, doc):
            results = {}
            for value, v_start, v_end in values:
                closest_year = None
                closest_distance = float('inf')
                for year, y_start, y_end in years:
                    distance = y_start - v_end
                    if distance >= 0 and distance < closest_distance:
                        closest_year = year
                        closest_distance = distance
                sign = self.analyse_context(doc, v_start)
                value = sign + value.text
                year_text = closest_year.text if closest_year else "NAN"
                if year_text not in results:
                    results[year_text] = []
                results[year_text].append(value)
            return results

    def analyse_context(self, doc, start):
        context_tokens = doc[max(start-25, 0):start]
        context_dict = {}
        for token in context_tokens:
            if token.lower_ in {"hausse", "augmentation", "baisse", "diminution", "appreciation", "excedent", "deficit","accroissement"}:
                context_dict[token.i] = token.lower_
        if context_dict:
            closest_key = min(context_dict.keys(), key=lambda x: abs(x-start))
            context_word = context_dict[closest_key]
            if context_word in {"hausse", "augmentation", "appreciation", "excedent","accroissement"}:
                return "+"
            elif context_word in {"baisse", "diminution", "deficit"}:
                return "-"
            else:
                return ""
        else:
            return ""



