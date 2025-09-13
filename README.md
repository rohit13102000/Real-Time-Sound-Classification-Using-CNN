# Real-Time-Sound-Classification-Using-CNN
This project involves building a Convolutional Neural Network (CNN) to classify environmental sounds using the UrbanSound8K dataset. Audio files are preprocessed into Mel spectrograms, which are treated as 2D image-like inputs to the CNN model. The model learns to identify various urban sound classes such as dog bark, siren, drilling, and more. The system achieves accurate classification by extracting time-frequency patterns from spectrograms and training the model using supervised learning.

#Deployment

It consists of the code (flask) and HTML (frontend) for project deployment on any cloud platform (AWS,GCP).

FILES:

cnn2.keras : Trained Model on Audios

labelencoder.pkl :Label Encoder

flask_model.py : Flask API
