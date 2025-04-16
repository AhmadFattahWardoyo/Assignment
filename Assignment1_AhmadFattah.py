#library
import librosa
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Definisi fungsi untuk memuat data audio dan melakukan fmcc serta zcr pada data audio
def extract_features(file_path='/content/drive/My Drive/Pattern Recognition/', max_pad_len=50):
    signal, sr = librosa.load(file_path, sr=16000)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    features = np.concatenate((mfcc.flatten(), zcr.flatten()))

    pad_width = max_pad_len - zcr.shape[1]
    if pad_width > 0:
        zcr = np.pad(zcr, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        zcr = zcr[:, :max_pad_len]
    return zcr.flatten()

# Load dataset dan ekstraksi fitur tiap dataset
animals = ['Bird', 'Cat', 'Dog']
x_data, y_data = [], []

data_dir = '/content/drive/My Drive/Pattern Recognition/'  # Folder yang berisi rekaman suara hewan
for idx, animal in enumerate(animals):
    animal_path = os.path.join(data_dir, animal)
    for file_name in os.listdir(animal_path):
        file_path = os.path.join(animal_path, file_name)
        features = extract_features(file_path)
        x_data.append(features)
        y_data.append(idx)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Training model
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=20)
model = KNeighborsClassifier(n_neighbors=4)
model.fit(x_train, y_train)

# Evaluasi model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model
with open('animal_sound_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('animal_sound_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Pengujian model dengan data asing
test_file = "/content/drive/My Drive/Pattern Recognition/data2.wav"
features = extract_features(test_file).reshape(1, -1)
prediction = model.predict(features)

print(f"Predicted Animal: {animals[prediction[0]]}")