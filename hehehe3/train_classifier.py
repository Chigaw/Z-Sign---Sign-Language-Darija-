import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Charger les données d'entraînement depuis le fichier pickle
data_dict = pickle.load(open('/Users/pc/hehehe3/data.pickle', 'rb'))

# Trouver la longueur maximale parmi tous les échantillons
max_length = max(len(sample) for sample in data_dict['data'])

# Remplir les échantillons pour atteindre la longueur maximale
data = np.array([sample + [0] * (max_length - len(sample)) for sample in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Ajuster le modèle RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')

# Entraîner le modèle
model.fit(x_train, y_train)

# Prédire les étiquettes sur l'ensemble de test
y_predict = model.predict(x_test)

# Calculer la précision
score = accuracy_score(y_predict, y_test)

# Afficher la précision
print('{}% of samples were classified correctly!'.format(score * 100))

# Enregistrer le modèle dans un fichier pickle
with open('/Users/pc/hehehe3/data.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)
