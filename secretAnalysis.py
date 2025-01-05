import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

"""
Bu projede, NLP modellerindeki gizli önyargıları tespit etmek, analiz etmek ve görselleştirmek amacıyla Crows-Pairs veri seti kullanarak kapsamlı bir çalışma gerçekleştirdim. 
Veri setini temizleyerek ön işleme adımlarını tamamladım ve eksik veri analizleri yaptım. Önyargı türlerinin dağılımını grafiklerle görselleştirerek detaylı bir analiz gerçekleştirdim. 
BERT modeliyle `fill-mask` yöntemi kullanarak cümle düzeyinde önyargı değerlendirmeleri yaptım ve modelin tahminlerini inceledim. 
Önyargılı (`sent_more`) ve önyargısız (`sent_less`) cümlelerde TF-IDF yöntemiyle kelime frekans analizleri yaparak öne çıkan kelimeleri belirledim ve Word Cloud görselleştirmeleri oluşturdum. 
Stereo/antistereo dağılımını heatmap kullanarak haritalandırdım ve PCA yöntemiyle kelime gömme uzayını 2 boyutlu olarak görselleştirdim. 
Ayrıca, önyargılı ve önyargısız cümlelerin uzunluklarını histogramlarla karşılaştırmalı olarak analiz ettim. 
Son olarak, elde ettiğim sonuçları temizlenmiş ve analiz edilmiş bir veri seti olarak kaydederek, NLP modellerindeki önyargıların anlaşılmasına ve azaltılmasına katkı sağlamayı hedefledim.
"""

# Veri Seti Yolu ve Çalışma Dizini
file_path = '/Users/bilge/Desktop/crows-pairs/data/crows_pairs_anonymized.csv'
os.chdir('/Users/bilge/Desktop/crows-pairs')
print("Çalışma dizini:", os.getcwd())

# Veri Setini Yükle
try:
    df = pd.read_csv(file_path)
    print(f"Toplam satır sayısı: {df.shape[0]}, Toplam sütun sayısı: {df.shape[1]}")
    print("Sütun isimleri:", df.columns)
except FileNotFoundError:
    print(f"Hata: Dosya bulunamadı -> {file_path}")
    exit()

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("\nVeri Seti İlk 5 Satır:")
print(df.head())

# Önyargı Türlerinin Dağılımı
bias_distribution = df['bias_type'].value_counts()
print("\n🔍 Önyargı Türlerinin Dağılımı:")
print(bias_distribution)

bias_distribution.plot(kind='bar')
plt.title('Önyargı Türlerinin Dağılımı')
plt.xlabel('Önyargı Türü')
plt.ylabel('Frekans')
plt.show()

print("\n Önyargılı Cümle Örnekleri:")
print(df['sent_more'].head(5))

print("\n Önyargısız Cümle Örnekleri:")
print(df['sent_less'].head(5))

# NLP Modeli (BERT) Yükle - Yerel Dizin
try:
    classifier = pipeline('fill-mask', model='/Users/bilge/.cache/huggingface/hub/models--bert-base-uncased')
    print("\n🤖 BERT Modeli Yerel Dizinden Yüklendi Başarıyla!")
except Exception as e:
    print(f"Hata: NLP modeli yüklenirken sorun oluştu. Detay: {e}")
    exit()

# Önyargılı ve Önyargısız Cümle Analizi
example_sent_more = df['sent_more'].iloc[0].replace("he", "[MASK]")
example_sent_less = df['sent_less'].iloc[0].replace("she", "[MASK]")

print("\n Önyargılı Cümle Analizi:")
result_more = classifier(example_sent_more)
for res in result_more[:5]:
    print(f"{res['sequence']} (Skor: {res['score']:.4f})")

print("\n Önyargısız Cümle Analizi:")
result_less = classifier(example_sent_less)
for res in result_less[:5]:
    print(f"{res['sequence']} (Skor: {res['score']:.4f})")

# Eksik Veri Analizi
print("\n Eksik Veri Analizi:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
df['sent_more'].str.len().plot(kind='hist', bins=30, alpha=0.7, label='Önyargılı Cümle')
df['sent_less'].str.len().plot(kind='hist', bins=30, alpha=0.7, label='Önyargısız Cümle')
plt.title('Önyargılı ve Önyargısız Cümle Uzunlukları Dağılımı')
plt.xlabel('Cümle Uzunluğu')
plt.ylabel('Frekans')
plt.legend()
plt.show()

# Önyargı Türlerinin Haritalandırılması
plt.figure(figsize=(12, 8))
sns.heatmap(pd.crosstab(df['bias_type'], df['stereo_antistereo']), annot=True, fmt='d', cmap='coolwarm')
plt.title('Önyargı Türleri ve Stereo/Antistereo Dağılımı')
plt.show()

# TF-IDF Kelime Analizi
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_sent_more = tfidf_vectorizer.fit_transform(df['sent_more'].fillna(''))
tfidf_sent_less = tfidf_vectorizer.fit_transform(df['sent_less'].fillna(''))

print("En çok geçen kelimeler (Önyargılı Cümleler):")
print(pd.DataFrame(tfidf_sent_more.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10))

print("En çok geçen kelimeler (Önyargısız Cümleler):")
print(pd.DataFrame(tfidf_sent_less.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10))

# PCA Görselleştirme
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(tfidf_sent_more.toarray())

plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, cmap='viridis')
plt.title('TF-IDF Kelime Uzayı PCA Görselleştirmesi')
plt.show()

# Temizlenmiş Veriyi Kaydet
output_file = '/Users/bilge/Desktop/crows-pairs/data/cleaned_crows_pairs.csv'
df.to_csv(output_file, index=False)
print(f"\n Temizlenmiş veri başarıyla kaydedildi: {output_file}")

print("\n Analiz başarıyla tamamlandı!")
