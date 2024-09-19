# classifier/views.py

import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from .forms import UploadImageForm
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm  # Importez votre formulaire personnalisé
from .forms import ProfileForm







# # Chemin du modèle pré-entraîné

# model_path = os.path.join(settings.BASE_DIR, 'classifier/trained_model/plant_disease_model.h5')

# # model_path = os.path.join(settings.BASE_DIR, 'classifier/trained_model/cnn_pvd.h5')

# model = tf.keras.models.load_model(model_path)


# Chemin du modèle pré-entraîné TFLite
model_path = os.path.join(settings.BASE_DIR, 'classifier/trained_model/plant_disease_model.tflite')

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Chemin vers les indices de classe
class_indices_path = os.path.join(settings.BASE_DIR, 'classifier/class_indices.json')
class_indices = json.load(open(class_indices_path))


# Traductions en Français des catégories
class_category = {
    "Apple___Apple_scab": "Pomme",
    "Apple___Black_rot": "Pomme",
    "Apple___Cedar_apple_rust": "Pomme",
    "Apple___healthy": "Pomme",
    "Blueberry___healthy": "Myrtille",
    "Cherry_(including_sour)___Powdery_mildew": "Cerise",
    "Cherry_(including_sour)___healthy": "Cerise",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maïs",
    "Corn_(maize)___Common_rust_": "Maïs",
    "Corn_(maize)___Northern_Leaf_Blight": "Maïs",
    "Corn_(maize)___healthy": "Maïs",
    "Grape___Black_rot": "Raisin",
    "Grape___Esca_(Black_Measles)": "Raisin",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Raisin",
    "Grape___healthy": "Raisin",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange",
    "Peach___Bacterial_spot": "Pêche",
    "Peach___healthy": "Pêche",
    "Pepper,_bell___Bacterial_spot": "Poivron",
    "Pepper,_bell___healthy": "Poivron",
    "Potato___Early_blight": "Pomme de terre",
    "Potato___Late_blight": "Pomme de terre",
    "Potato___healthy": "Pomme de terre",
    "Raspberry___healthy": "Framboise",
    "Soybean___healthy": "Soja",
    "Squash___Powdery_mildew": "Courge",
    "Strawberry___Leaf_scorch": "Fraise",
    "Strawberry___healthy": "Fraise",
    "Tomato___Bacterial_spot": "Tomate",
    "Tomato___Early_blight": "Tomate",
    "Tomato___Late_blight": "Tomate",
    "Tomato___Leaf_Mold": "Tomate",
    "Tomato___Septoria_leaf_spot": "Tomate",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomate",
    "Tomato___Target_Spot": "Tomate",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomate",
    "Tomato___Tomato_mosaic_virus": "Tomate",
    "Tomato___healthy": "Tomate"
}

# Plantes Saines
healthy_plants = {
    "Apple___healthy",
    "Blueberry___healthy",
    "Corn_(maize)___healthy",
    "Grape___healthy",
    "Peach___healthy",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Strawberry___healthy",
    "Tomato___healthy"
}

# Descriptions des Classes
class_descriptions = {
    "Apple___Apple_scab": "Pour traiter la Tavelure du pommier, utilisez des fongicides spécifiques au printemps avant la floraison et à l'automne après la récolte. Assurez-vous de retirer et de brûler les feuilles infectées pour réduire la propagation.",
    "Apple___Black_rot": "Contre la Pourriture noire du pommier, utilisez des fongicides recommandés dès l'apparition des premiers symptômes. Pratiquez une bonne aération des arbres et éliminez les fruits infectés pour limiter la propagation.",
    "Apple___Cedar_apple_rust": "Pour lutter contre la Rouille du pommier, utilisez des fongicides dès l'apparition des premiers signes. Éliminez les feuilles infectées et pratiquez une rotation des cultures pour réduire la propagation.",
    "Apple___healthy": "Pas besoin de traitement, la Pomme est en bonne santé !",
    "Blueberry___healthy": "Pas besoin de traitement, la Myrtille est en bonne santé !",
    "Cherry_(including_sour)___Powdery_mildew": "Pour traiter l'Oïdium sur les cerises, utilisez des fongicides dès l'apparition des symptômes. Assurez-vous d'aérer les arbres pour réduire l'humidité et évitez les arrosages excessifs.",
    "Cherry_(including_sour)___healthy": "Pas besoin de traitement, la Cerise est en bonne santé !",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Pour traiter la Tache cercosporéenne et la Tache grise des feuilles de maïs, utilisez des fongicides adaptés dès l'apparition des premiers signes. Pratiquez une rotation des cultures et éliminez les débris végétaux infectés pour limiter la propagation.",
    "Corn_(maize)___Common_rust_": "Pour traiter la Rouille commune du maïs, utilisez des variétés résistantes lorsque possible et appliquez des fongicides dès l'apparition des premiers signes. Pratiquez une bonne gestion des résidus de culture pour réduire la propagation.",
    "Corn_(maize)___Northern_Leaf_Blight": "Pour traiter la Brûlure des feuilles du nord du maïs, utilisez des fongicides dès l'apparition des symptômes et assurez-vous d'une bonne gestion de l'irrigation pour éviter l'humidité excessive.",
    "Corn_(maize)___healthy": "Pas besoin de traitement, le Maïs est en bonne santé !",
    "Grape___Black_rot": "Pour traiter la Pourriture noire du raisin, utilisez des fongicides adaptés dès l'apparition des premiers signes de la maladie. Assurez-vous d'une bonne circulation de l'air autour des grappes et pratiquez une taille appropriée pour réduire l'humidité et limiter la propagation.",
    "Grape___Esca_(Black_Measles)": "Pour traiter la Rougeole noire (Raisin Elsa), utilisez des fongicides dès l'apparition des premiers symptômes. Éliminez les grappes infectées et assurez-vous d'une bonne aération pour réduire l'humidité.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Pour traiter la Tache des feuilles de vigne (Tache foliaire d'Isariopsis), utilisez des fongicides recommandés dès l'apparition des symptômes. Pratiquez une bonne aération des vignes et enlevez les feuilles infectées pour limiter la propagation.",
    "Grape___healthy": "Pas besoin de traitement, le Raisin est en bonne santé !",
    "Orange___Haunglongbing_(Citrus_greening)": "Pour traiter le Verdissement des agrumes (Huanglongbing sur l'orange), utilisez des méthodes de gestion intégrée des ravageurs et des maladies. Supprimez et détruisez les arbres infectés, utilisez des insecticides pour contrôler les vecteurs et pratiquez une nutrition équilibrée des plantes pour renforcer leur résistance.",
    "Peach___Bacterial_spot": "Pour traiter la Tache bactérienne de la pêche, utilisez des fongicides ou des antibiotiques recommandés dès l'apparition des symptômes. Pratiquez une bonne gestion de l'irrigation pour éviter l'humidité excessive et éliminez les parties de l'arbre infectées pour limiter la propagation.",
    "Peach___healthy": "Pas besoin de traitement, la Pêche est en bonne santé !",
    "Pepper,_bell___Bacterial_spot": "Pour traiter la Tache bactérienne du poivron, utilisez des fongicides ou des solutions à base d'oxychlorure de cuivre dès l'apparition des premiers symptômes. Assurez-vous d'une bonne rotation des cultures et éliminez les plants infectés pour limiter la propagation de la maladie.",
    "Pepper,_bell___healthy": "Pas besoin de traitement, le Poivron est en bonne santé !",
    "Potato___Early_blight": "Pour traiter le Mildiou précoce de la pomme de terre, utilisez des fongicides recommandés dès l'apparition des premiers signes de la maladie. Pratiquez une rotation des cultures, évitez l'irrigation excessive et enlevez les parties de plantes infectées pour limiter la propagation.",
    "Potato___Late_blight": "Pour traiter le Mildiou tardif de la pomme de terre, utilisez des fongicides spécifiques dès l'apparition des premiers symptômes. Assurez-vous d'une bonne gestion de l'irrigation pour éviter l'humidité excessive et pratiquez une rotation des cultures pour réduire la propagation de la maladie.",
    "Potato___healthy": "Pas besoin de traitement, la Pomme de terre est en bonne santé !",
    "Raspberry___healthy": "Pas besoin de traitement, la Framboise est en bonne santé !",
    "Soybean___healthy": "Pas besoin de traitement, le Soja est en bonne santé !",
    "Squash___Powdery_mildew": "Pour traiter l'Oïdium de la courge, utilisez des fongicides adaptés dès l'apparition des premiers signes de la maladie. Assurez-vous d'une bonne aération des plantes et évitez les arrosages excessifs pour réduire l'humidité.",
    "Strawberry___Leaf_scorch": "Pour traiter la Brûlure des feuilles de fraisier, utilisez des fongicides appropriés dès l'apparition des premiers symptômes. Maintenez une bonne aération autour des plants et évitez les arrosages excessifs pour limiter la propagation de la maladie.",
    "Strawberry___healthy": "Pas besoin de traitement, la Fraise est en bonne santé !",
    "Tomato___Bacterial_spot": "Pour traiter la Tache bactérienne de la tomate, utilisez des fongicides ou des solutions à base de cuivre dès les premiers signes de la maladie. Évitez l'arrosage par aspersion pour limiter la propagation des bactéries et enlevez les parties infectées des plantes pour réduire la contamination.",
    "Tomato___Early_blight": "Pour traiter le Mildiou précoce de la tomate, utilisez des fongicides dès l'apparition des premiers symptômes. Assurez-vous d'une bonne aération autour des plants et évitez les arrosages excessifs pour réduire l'humidité, ce qui favorise la propagation de la maladie.",
    "Tomato___Late_blight": "Pour traiter le Mildiou tardif de la tomate, utilisez des fongicides spécifiques dès l'apparition des premiers signes de la maladie. Pratiquez une bonne gestion de l'irrigation pour éviter l'humidité excessive et enlevez les parties de plantes infectées pour limiter la propagation.",
    "Tomato___Leaf_Mold": "Pour traiter la moisissure des feuilles de tomate, utilisez des fongicides adaptés dès l'apparition des premiers symptômes. Maintenez une bonne circulation d'air autour des plants et évitez l'humidité excessive en ajustant l'arrosage.",
    "Tomato___Septoria_leaf_spot": "Pour traiter la Tache septorienne sur les feuilles de la tomate, utilisez des fongicides appropriés dès l'apparition des premiers symptômes. Pratiquez une rotation des cultures et assurez-vous d'une bonne gestion de l'humidité pour limiter la propagation de la maladie.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Pour traiter les tétranyques rouges, utilisez des acaricides efficaces dès l'apparition des premiers signes d'infestation. Assurez-vous d'une bonne gestion de l'humidité et de la ventilation pour contrôler leur propagation sur les plantes.",
    "Tomato___Target_Spot": "Pour traiter la Tache concentrique sur les tomates, utilisez des fongicides dès l'apparition des premiers symptômes. Pratiquez une rotation des cultures, enlevez les feuilles infectées, et assurez-vous d'une bonne aération pour limiter la propagation de la maladie.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Pour traiter le virus de la feuille jaune en cuillère de la tomate, utilisez des pratiques de gestion intégrée des maladies comme l'utilisation de plants résistants, l'élimination des plantes infectées, et le contrôle des insectes vecteurs.",
    "Tomato___Tomato_mosaic_virus": "Pour traiter le virus de la mosaïque de la tomate, utilisez des plants résistants si disponibles et assurez-vous de contrôler les insectes vecteurs comme les pucerons. Éliminez les plantes infectées dès leur identification pour limiter la propagation du virus dans le jardin.",
    "Tomato___healthy": "Pas besoin de traitement, la Tomate est en bonne santé !"
}

# Traductions des Maladies des Classes
class_translations = {
    "Apple___Apple_scab": "Tavelure du pommier",
    "Apple___Black_rot": "Pourriture noire du pommier",
    "Apple___Cedar_apple_rust": "Rouille du pommier, cèdre",
    "Apple___healthy": "Pommier en bonne santé",
    "Blueberry___healthy": "Myrtille en bonne santé",
    "Cherry_(including_sour)___Powdery_mildew": "Cerise (y compris acide) Oïdium",
    "Cherry_(including_sour)___healthy": "Cerise en bonne santé",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Tache cercosporéenne des feuilles de maïs",
    "Corn_(maize)___Common_rust_": "Rouille commune du maïs",
    "Corn_(maize)___Northern_Leaf_Blight": "Brûlure des feuilles du nord du maïs",
    "Corn_(maize)___healthy": "Maïs en bonne santé",
    "Grape___Black_rot": "Pourriture noire du raisin",
    "Grape___Esca_(Black_Measles)": "Raisin Elsa (rougeole noire)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Tache des feuilles de vigne",
    "Grape___healthy": "Raisin en bonne santé",
    "Orange___Haunglongbing_(Citrus_greening)": "Verdissement des agrumes de Huanglongbing",
    "Peach___Bacterial_spot": "Tache bactérienne de la pêche",
    "Peach___healthy": "Pêche en bonne santé",
    "Pepper,_bell___Bacterial_spot": "Tache bactérienne du poivron",
    "Pepper,_bell___healthy": "Poivron en bonne santé",
    "Potato___Early_blight": "Mildiou précoce de la pomme de terre",
    "Potato___Late_blight": "Mildiou tardif de la pomme de terre",
    "Potato___healthy": "Pomme de terre en bonne santé",
    "Raspberry___healthy": "Framboise en bonne santé",
    "Soybean___healthy": "Soja en bonne santé",
    "Squash___Powdery_mildew": "Oïdium de la courge",
    "Strawberry___Leaf_scorch": "Brûlure des feuilles de fraisier",
    "Strawberry___healthy": "Fraise en bonne santé",
    "Tomato___Bacterial_spot": "Tache bactérienne de la tomate",
    "Tomato___Early_blight": "Mildiou précoce de la tomate",
    "Tomato___Late_blight": "Mildiou tardif de la tomate",
    "Tomato___Leaf_Mold": "Moisissure des feuilles de tomates",
    "Tomato___Septoria_leaf_spot": "Tache septorienne sur les feuilles de la tomate",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tétranyques rouges",
    "Tomato___Target_Spot": "Tache concentrique tomate",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus de la feuille jaune en cuillère de la tomate",
    "Tomato___Tomato_mosaic_virus": "Virus de la mosaïque de la tomate",
    "Tomato___healthy": "Tomate en bonne santé"
}

# Fonction pour charger et prétraiter l'image avec Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Fonction pour prédire la classe d'une image en utilsant le modèle CNN 
# def predict_image_class(image_path):
#     preprocessed_img = load_and_preprocess_image(image_path)
#     predictions = model.predict(preprocessed_img)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class_name = class_indices[str(predicted_class_index)]
#     return predicted_class_name

# Fonction pour prédire la classe d'une image en utilisant le modèle TFLite
def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Assigner les données d'entrée au modèle
    interpreter.set_tensor(input_details['index'], preprocessed_img)
    interpreter.invoke()

    # Obtenir les résultats de la prédiction
    predictions = interpreter.get_tensor(output_details['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Vue pour gérer le téléchargement d'image et la classification
@login_required
def classify_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']
            temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
            
            with open(temp_image_path, 'wb') as f:
                for chunk in uploaded_image.chunks():
                    f.write(chunk)

            prediction = predict_image_class(temp_image_path)
            
            translation = class_translations.get(prediction, prediction.replace('_', ' '))
            description = class_descriptions.get(prediction, 'Description non disponible')
            category = class_category.get(prediction, 'Catégorie non disponible')

            # Check if the prediction is in healthy plants
            is_healthy = prediction in healthy_plants

            # Passer l'URL de l'image téléchargée au contexte du modèle
            uploaded_image_url = os.path.join(settings.MEDIA_URL, 'temp_image.jpg')

            return render(request, 'classifier/classify_image.html', {
                'prediction': translation,
                'description': description,
                'category': category,
                'uploaded_image_url': uploaded_image_url,
                'is_healthy': is_healthy,
            })
    else:
        form = UploadImageForm()

    return render(request, 'classifier/classify_image.html', {'form': form})


@login_required
def accueil(request):
    # Vérifie si l'utilisateur est authentifié
    if request.user.is_authenticated:
        username = request.user.username
    else:
        username = None  # Ou une valeur par défaut si l'utilisateur n'est pas authentifié

    # Contexte à passer au template
    context = {
        'username': username
    }

    # Rendu du template avec le contexte
    return render(request, 'classifier/accueil.html', context)

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout

def register(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Log in the user after signing up
            login(request, user)
            return redirect('accueil')  # Redirigez l'utilisateur vers la page d'accueil après l'inscription
    else:
        form = SignUpForm()

    return render(request, 'user/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect('accueil')  # Rediriger vers la page d'accueil après connexion
    else:
        form = AuthenticationForm()
    return render(request, 'user/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('login')  # Rediriger vers la page d'accueil après déconnexion

@login_required
def profile(request):
    user = request.user
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('profile')  # Redirigez vers la même page de profil après la mise à jour
    else:
        form = ProfileForm(instance=user)

    return render(request, 'user/profile.html', {'form': form})
