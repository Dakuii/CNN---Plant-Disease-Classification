# forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Obligatoire. Indiquez une adresse email valide.')

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Cet email est déjà utilisé. Veuillez choisir un autre email.")
        return email

class UploadImageForm(forms.Form):
    image = forms.ImageField()

class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email']
