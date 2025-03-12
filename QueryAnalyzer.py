import openai
from sentence_transformers import SentenceTransformer
from typing import Dict
from dotenv import load_dotenv
import os 


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class FinanceQueryAnalyzer:
    def __init__(self):
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Intent kategorilerini tanımla
        self.labels = [
            "yatırım rehberi",  # (Örn: Hangi hisse senetleri daha güvenli?)
            "piyasa analizi",  # (Örn: Borsa endeksi neden düştü?)
            "finansal okur yazarlık",  # (Örn: Bitcoin 2025'te ne olur?)
            "tasarruf ve bütçeleme",  # (Örn: Aylık gelirimi nasıl yönetebilirim?)
            "ekonomi politikası"  # (Örn: Merkez Bankası faiz artırır mı?)
        ]

    def get_temperature(self, intent: str) -> float:
        """Her intent için uygun temperature değerini belirler."""
        temperature_settings = {
            "yatırım rehberi": 0.4,
            "piyasa analizi": 0.3,
            "finansal okur yazarlık": 0.6,
            "tasarruf ve bütçeleme": 0.5,
            "ekonomi politikası": 0.2
        }
        return temperature_settings.get(intent, 0.5) 

    def analyze_query_intent(self, query: str) -> Dict:
        prompt = f"""
        Aşağıdaki sorgunun amacını belirleme:
        
        Sorgu: "{query}"
        
        Bu sorgu aşağıdaki kategorilerden hangisine en çok uyuyor? Sadece kategori adını yaz.
        
        Kategoriler:
        {", ".join(self.labels)}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        intent_text = response['choices'][0]['message']['content'].strip().lower()
        
        matched_intent = None
        for label in self.labels:
            if label.lower() in intent_text:
                matched_intent = label
                break
        
        intent = matched_intent if matched_intent else self.labels[0]
        
        confidence = 1.0 if matched_intent else 0.7
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_intents": {intent: confidence}
        }

    def expand_query(self, query: str) -> str:
        intent_result = self.analyze_query_intent(query)
        intent = intent_result["intent"]
        temperature = self.get_temperature(intent)
        
        templates = {
            "yatırım rehberi": f"""
            BU BİR SORGU GENİŞLETME GÖREVİDİR.
            
            GÖREV: Aşağıdaki yatırım sorusunu profesyonel bir yatırımcı bakış açısıyla genişlet.
            
            KURALLAR:
            1. SONUÇ MUTLAKA SORU FORMATI OLMALIDIR (soru işareti ile bitmeli)
            2. Sadece genişletilmiş soruyu yaz, açıklama veya cevap verme
            3. Risk, getiri, likidite, vade ve volatilite kavramlarını dahil et
            4. Finansal terimleri doğru kullan, terimsel hataları düzelt
            5. Maksimum 200 token kullan
            
            SORGU: {query}
            
            GENİŞLETİLMİŞ SORU: 
            """,
            
            "piyasa analizi": f"""
            BU BİR SORGU GENİŞLETME GÖREVİDİR.
            
            GÖREV: Aşağıdaki piyasa sorusunu profesyonel bir analist bakış açısıyla genişlet.
            
            KURALLAR:
            1. SONUÇ MUTLAKA SORU FORMATI OLMALIDIR (soru işareti ile bitmeli)
            2. Sadece genişletilmiş soruyu yaz, açıklama veya cevap verme
            3. Teknik ve temel analiz kavramlarını dahil et
            4. Trend, momentum, destek/direnç seviyelerinden bahset
            5. Endüstri dinamikleri ve makroekonomik faktörleri ekle
            6. Maksimum 200 token kullan
            
            SORGU: {query}
            
            GENİŞLETİLMİŞ SORU:
            """,
            
            "finansal okur yazarlık": f"""
            BU BİR SORGU GENİŞLETME GÖREVİDİR.
            
            GÖREV: Aşağıdaki finansal okuryazarlık sorusunu eğitici bir yaklaşımla genişlet.
            
            KURALLAR:
            1. SONUÇ MUTLAKA SORU FORMATI OLMALIDIR (soru işareti ile bitmeli)
            2. Sadece genişletilmiş soruyu yaz, açıklama veya cevap verme
            3. Karmaşık finans terimlerini basitleştir, günlük dil kullan
            4. Temel finans kavramlarını (bütçe, tasarruf, yatırım, borç) dahil et
            5. Finansal eğitim odaklı ve anlaşılır olmalı
            6. Maksimum 200 token kullan

            SORGU: {query}
            
            GENİŞLETİLMİŞ SORU:
            """,
            
            "tasarruf ve bütçeleme": f"""
            BU BİR SORGU GENİŞLETME GÖREVİDİR.
            
            GÖREV: Aşağıdaki tasarruf/bütçe sorusunu bireysel finans perspektifiyle genişlet.
            
            KURALLAR:
            1. SONUÇ MUTLAKA SORU FORMATI OLMALIDIR (soru işareti ile bitmeli)
            2. Sadece genişletilmiş soruyu yaz, açıklama veya cevap verme
            3. Nakit akışı, acil durum fonu, harcama takibi kavramlarını dahil et
            4. 50/30/20 bütçe kuralı veya sıfır-tabanlı bütçeleme gibi tekniklere atıf yap
            5. Uzun vadeli finansal hedeflerle ilişkilendir
            6. Maksimum 200 token kullan
            
            SORGU: {query}
            
            GENİŞLETİLMİŞ SORU:
            """,
            
            "ekonomi politikası": f"""
            BU BİR SORGU GENİŞLETME GÖREVİDİR.
            
            GÖREV: Aşağıdaki ekonomi politikası sorusunu makroekonomik bakış açısıyla genişlet.
            
            KURALLAR:
            1. SONUÇ MUTLAKA SORU FORMATI OLMALIDIR (soru işareti ile bitmeli)
            2. Sadece genişletilmiş soruyu yaz, açıklama veya cevap verme
            3. Enflasyon, faiz oranları, para politikası ve maliye politikası kavramlarını dahil et
            4. Merkez bankası kararları ve ekonomik göstergelere atıf yap
            5. Teknik bir dil kullan ve akademik bir üslup benimse
            6. Maksimum 200 token kullan
            SORGU: {query}
            
            GENİŞLETİLMİŞ SORU:
            """
        }
        
        selected_template = templates.get(intent)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": selected_template}],
            max_tokens=200,
            temperature=temperature
        )
        
        return response['choices'][0]['message']['content'].strip()

    def embed_query(self, query: str):
        """Sorguyu finansal anlamda vektörleştirerek temsil eder."""
        embedding = self.embedder.encode(query)
        return embedding  


if __name__ == "__main__":
    analyzer = FinanceQueryAnalyzer()

    query = "Merkez banakası neye göre faizi arttırır?" 

    print(f"\n--- Sorgu: {query} ---")
    intent_result = analyzer.analyze_query_intent(query)
    print("Sorgu Amacı:", intent_result["intent"], f"(Güven: {intent_result['confidence']})")
    print("Seçilen Sıcaklık:", analyzer.get_temperature(intent_result["intent"]))
    
    expanded_query = analyzer.expand_query(query)
    print("Genişletilmiş Sorgu:", expanded_query)
    
    vector = analyzer.embed_query(query)
    print("Vektör Temsili (ilk 3 boyut):", vector[:3])
    print("-" * 50)