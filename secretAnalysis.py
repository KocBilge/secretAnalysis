import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap
import warnings

# Paralel işlem uyarısını devre dışı bırakma
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uyarıları yoksayma
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

"""
Bu projede, NLP modellerindeki gizli önyargıları tespit etmek, analiz etmek ve görselleştirmek amacıyla kapsamlı bir çalışma gerçekleştirildi.
"""

# Veri Seti Yolu ve Çalışma Dizini
file_path = '/Users/bilge/Downloads/Dataset.xlsx'
os.chdir('/Users/bilge/Desktop')

try:
    df = pd.read_excel(file_path)
    print(f"Toplam satır sayısı: {df.shape[0]}, Toplam sütun sayısı: {df.shape[1]}")
    print("Sütun isimleri:", df.columns)
except FileNotFoundError:
    print(f"Hata: Dosya bulunamadı -> {file_path}")
    exit()

print("\nVeri Seti İlk 5 Satır:")
print(df.head())

# Eksik Veri Analizi
print("\nEksik Veri Analizi:")
print(df.isnull().sum())

# Gönderilerin uzunluklarının histogramını çizme
plt.figure(figsize=(10, 6))
df['Post description'].str.len().plot(kind='hist', bins=30, alpha=0.7)
plt.title('Gönderi Uzunlukları Dağılımı')
plt.xlabel('Cümle Uzunluğu')
plt.ylabel('Frekans')
plt.show()

# Sentiment Dağılımı
plt.figure(figsize=(10, 6))
df['Sentiment'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Sentiment Dağılımı')
plt.xlabel('Duygu Durumu')
plt.ylabel('Frekans')
plt.show()

# Stress ve Anksiyete Durumu Dağılımı
plt.figure(figsize=(10, 6))
df['Stress or Anxiety'].value_counts().plot(kind='bar', color='orange')
plt.title('Stres ve Anksiyete Durumu Dağılımı')
plt.xlabel('Stres/Anksiyete Durumu')
plt.ylabel('Frekans')
plt.show()

# DistilBERT Modeli Yükleme
try:
    classifier = pipeline('fill-mask', model='distilbert-base-uncased')
    print("\n🤖 DistilBERT Modeli Yüklendi Başarıyla!")
except Exception as e:
    print(f"Hata: NLP modeli yüklenirken sorun oluştu. Detay: {e}")
    exit()

# Önyargılı ve Önyargısız Cümle Analizi
example_sent_more = df['Post description'].iloc[0].replace("he", "[MASK]")
example_sent_less = df['Post description'].iloc[1].replace("she", "[MASK]")

# Önyargılı Cümlede [MASK] Kontrolü
if "[MASK]" not in example_sent_more:
    print("Hata: Önyargılı cümlede [MASK] tokeni bulunamadı.")
    print("Cümle:", example_sent_more)
else:
    print("\nÖnyargılı Cümle Analizi:")
    result_more = classifier(example_sent_more)
    print("Önyargılı Cümle Çıktısı:", result_more)
    if isinstance(result_more, list) and all(isinstance(res, dict) for res in result_more):
        for res in result_more:
            print(f"Prediction: {res['sequence']} with a score of {res['score']}")
    else:
        print("Çıktı beklenmeyen formatta:", result_more)

# Önyargısız Cümlede [MASK] Kontrolü
if "[MASK]" not in example_sent_less:
    print("Hata: Önyargısız cümlede [MASK] tokeni bulunamadı.")
    print("Cümle:", example_sent_less)
else:
    print("\nÖnyargısız Cümle Analizi:")
    result_less = classifier(example_sent_less)
    print("Önyargısız Cümle Çıktısı:", result_less)
    if isinstance(result_less, list) and all(isinstance(res, dict) for res in result_less):
        for res in result_less:
            print(f"Prediction: {res['sequence']} with a score of {res['score']}")
    else:
        print("Çıktı beklenmeyen formatta:", result_less)

# TF-IDF Kelime Frekans Analizi
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_sent_more = tfidf_vectorizer.fit_transform(df['Post description'].fillna(''))
print("En çok geçen kelimeler (Önyargılı Cümleler):")
print(pd.DataFrame(tfidf_sent_more.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10))

# PCA Görselleştirme
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(tfidf_sent_more.toarray())

plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
plt.title('TF-IDF Kelime Uzayı PCA Görselleştirmesi')
plt.show()

# T-SNE Görselleştirme
print("\nT-SNE Analizi Başlıyor...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
tsne_results = tsne.fit_transform(tfidf_sent_more.toarray())

colors = np.random.rand(tsne_results.shape[0])
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='viridis', alpha=0.7)
plt.title("T-SNE ile Embedding Uzayı Görselleştirmesi")
plt.xlabel("T-SNE Bileşeni 1")
plt.ylabel("T-SNE Bileşeni 2")
plt.show()

# UMAP Görselleştirme
print("\nUMAP Analizi Başlıyor...")
umap_reducer = umap.UMAP(n_components=2)
umap_results = umap_reducer.fit_transform(tfidf_sent_more.toarray())

colors = np.random.rand(umap_results.shape[0])
plt.figure(figsize=(10, 8))
plt.scatter(umap_results[:, 0], umap_results[:, 1], c=colors, cmap='viridis', alpha=0.7)
plt.title("UMAP ile Embedding Uzayı Görselleştirmesi")
plt.xlabel("UMAP Bileşeni 1")
plt.ylabel("UMAP Bileşeni 2")
plt.show()

# Temizlenmiş Veriyi Kaydet
output_file = '/Users/bilge/Downloads/Cleaned_Dataset.csv'
df.to_csv(output_file, index=False)
print(f"\nTemizlenmiş veri başarıyla kaydedildi: {output_file}")

print("\nAnaliz başarıyla tamamlandı!")
