import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import os
import lime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap
import warnings
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from lime.lime_text import LimeTextExplainer
from nlpaug.augmenter.word import SynonymAug
import seaborn as sns

# Paralel işlem uyarısını devre dışı bırakma
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uyarıları yoksayma
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

"""
Bu projede, NLP modellerindeki gizli önyargıları tespit etmek, analiz etmek ve görselleştirmek amacıyla kapsamlı bir çalışma gerçekleştirildi.
"""

# Veri Setini Yükleme
try:
    dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
    df = pd.DataFrame(dataset['train'])  # Eğitim verisini DataFrame'e dönüştürüyoruz

    print(f"Toplam satır sayısı: {df.shape[0]}, Toplam sütun sayısı: {df.shape[1]}")
    print("Sütun isimleri:", df.columns)

    # 'text' sütunu kontrolü
    if 'text' not in df.columns:
        print("Hata: 'text' sütunu veri setinde bulunamadı. Mevcut sütunlar:", df.columns)
        exit()

    # UTF-8 hatalarını temizleme
    df['text'] = df['text'].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else '')

except Exception as e:
    print(f"Hata: Veri seti yüklenemedi. Detay: {e}")
    exit()

print("\nVeri Seti İlk 5 Satır:")
print(df.head())

# Eksik Veri Analizi
print("\nEksik Veri Analizi:")
print(df.isnull().sum())

# Gönderilerin uzunluklarının histogramını çizme
plt.figure(figsize=(10, 6))
df['text'].str.len().plot(kind='hist', bins=30, alpha=0.7)
plt.title('Gönderi Uzunlukları Dağılımı')
plt.xlabel('Cümle Uzunluğu')
plt.ylabel('Frekans')
plt.show()

# Sentiment Dağılımı
plt.figure(figsize=(10, 6))
df['label'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Sentiment Dağılımı')
plt.xlabel('Duygu Durumu')
plt.ylabel('Frekans')
plt.show()

# DistilBERT Modeli Yükleme
try:
    classifier = pipeline('text-classification', model='distilbert-base-uncased')
    print("\n\ud83e\udd16 DistilBERT Modeli Yüklendi Başarıyla!")
except Exception as e:
    print(f"Hata: NLP modeli yüklenirken sorun oluştu. Detay: {e}")
    exit()

# Model Performansının Değerlendirilmesi
try:
    y_true = df['label']
    y_pred = [classifier(text)[0]['label'] for text in df['text']]

    print("\nModel Performansı:")
    print(classification_report(y_true, y_pred))

    # Karışıklık Matrisi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Karışıklık Matrisi')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.show()

except Exception as e:
    print(f"Hata: Model değerlendirme sırasında sorun oluştu. Detay: {e}")
    exit()

# LIME ile Model Açıklanabilirliği
try:
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    explanation = explainer.explain_instance(
        df['text'].iloc[0],
        lambda x: [classifier(text)[0]['score'] for text in x]
    )
    explanation.show_in_notebook()
except Exception as e:
    print(f"Hata: LIME açıklaması oluşturulamadı. Detay: {e}")

# Veri Zenginleştirme (Data Augmentation)
try:
    aug = SynonymAug()
    df['augmented_text'] = df['text'].apply(lambda x: aug.augment(x))
    print("\nVeri Zenginleştirme Tamamlandı. Örnek Zenginleştirilmiş Metinler:")
    print(df[['text', 'augmented_text']].head())
except Exception as e:
    print(f"Hata: Veri zenginleştirme sırasında sorun oluştu. Detay: {e}")

# Türkçe stop words kelimeleri
turkish_stop_words = [
    "ve", "bir", "bu", "da", "de", "için", "ile", "ama", "eğer", "daha", "çok", "gibi", "ancak", "ise",
    "diye", "ki", "şu", "çünkü", "o", "kadar", "ne", "mu", "mi", "mı", "biraz", "bazı", "her", "tüm", "bazıları"
]

# TF-IDF Kelime Frekans Analizi
tfidf_vectorizer = TfidfVectorizer(stop_words=turkish_stop_words, max_features=50)
tfidf_sent_more = tfidf_vectorizer.fit_transform(df['text'].fillna(''))
print("En çok geçen kelimeler:")
print(pd.DataFrame(tfidf_sent_more.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10))

# PCA Görselleştirme
try:
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(tfidf_sent_more.toarray())

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.title('TF-IDF Kelime Uzayı PCA Görselleştirmesi')
    plt.show()
except Exception as e:
    print(f"Hata: PCA görselleştirme sırasında sorun oluştu. Detay: {e}")

# T-SNE Görselleştirme
try:
    print("\nT-SNE Analizi Başlıyor...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(tfidf_sent_more.toarray())

    colors = np.random.rand(tsne_results.shape[0])
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='viridis', alpha=0.7)
    plt.title("T-SNE ile Embedding Uzayı Görselleştirmesi")
    plt.xlabel("T-SNE Bileşeni 1")
    plt.ylabel("T-SNE Bileşeni 2")
    plt.show()
except Exception as e:
    print(f"Hata: T-SNE görselleştirme sırasında sorun oluştu. Detay: {e}")

# UMAP Görselleştirme
try:
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
except Exception as e:
    print(f"Hata: UMAP görselleştirme sırasında sorun oluştu. Detay: {e}")

# Temizlenmiş Veriyi Kaydet
output_file = '/Users/bilge/Downloads/Cleaned_Dataset.csv'
try:
    df.to_csv(output_file, index=False)
    print(f"\nTemizlenmiş veri başarıyla kaydedildi: {output_file}")
except Exception as e:
    print(f"Hata: Temizlenmiş veri kaydedilirken sorun oluştu. Detay: {e}")

print("\nAnaliz başarıyla tamamlandı!")
