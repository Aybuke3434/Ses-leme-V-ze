import numpy as np
from glob import glob
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ses dosyalarından MFCC özelliklerini çıkaran fonksiyon
def extract_features(file_path):
    rate, audio = wavfile.read(file_path)
    mfcc_features = mfcc(audio, rate, numcep=13)  # MFCC özelliklerini çıkarma
    return mfcc_features.flatten()

# Veri setini yükleme fonksiyonu
def load_data(data_path):
    features = []
    labels = []
    for class_dir in glob(data_path + "/*"):
        for audio_file in glob(class_dir + "/*.wav"):
            features.append(extract_features(audio_file))
            labels.append(class_dir.split("/")[-1])
    return np.array(features), np.array(labels)

# Veri setinin yolu
data_path = "../data/TurEV-DB-master/TurEV-DB-master/Sound Source/"
X, y = load_data(data_path)

# Etiketleri encode etme işlemi
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standart ölçeklendirme işlemi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression modeli oluşturma
model = LogisticRegression(max_iter=1000)

# Hiper-parametre ızgarasını tanımlama
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Düzenleme parametresi
    'solver': ['newton-cg', 'lbfgs', 'liblinear']  # Çözümleyiciler
}

# Grid Search ile hiper-parametre optimizasyonu
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi parametreleri ve en iyi skoru yazdırma
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor (doğruluk):", grid_search.best_score_)

# En iyi model ile eğitim ve test doğruluklarını değerlendirme
best_model = grid_search.best_estimator_
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)

print("Eğitim doğruluğu:", train_accuracy)
print("Test doğruluğu:", test_accuracy)
