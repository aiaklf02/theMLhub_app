# app/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Utilisateur

class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True)
    username = forms.CharField(max_length=150, required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True, label='Password')
    profile_picture_path = forms.FileField(required=False)
    country = forms.CharField(max_length=100, required=False)
    full_name = forms.CharField(max_length=150, required=False)
    bio = forms.CharField(widget=forms.Textarea, required=False)
    mobile_number = forms.CharField(max_length=15, required=False)

    class Meta:
        model = Utilisateur
        fields = ['username', 'email', 'password', 'profile_picture_path', 'country', 'full_name', 'bio', 'mobile_number']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'password': forms.PasswordInput(attrs={'class': 'form-control'}),
            'profile_picture_path': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'country': forms.TextInput(attrs={'class': 'form-control'}),
            'full_name': forms.TextInput(attrs={'class': 'form-control'}),
            'bio': forms.Textarea(attrs={'class': 'form-control'}),
            'mobile_number': forms.TextInput(attrs={'class': 'form-control'}),
        }
    def clean_password(self):
        password = self.cleaned_data.get('password')
        # Ensure both password1 and password2 match
        if password:
            self.cleaned_data['password1'] = password
            self.cleaned_data['password2'] = password
        return password

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.profile_picture_path = self.cleaned_data['profile_picture_path']
        user.country = self.cleaned_data['country']
        user.full_name = self.cleaned_data['full_name']
        user.bio = self.cleaned_data.get('bio', '')  # Ensure empty bio works
        user.mobile_number = self.cleaned_data.get('mobile_number', '')  # Ensure empty mobile works
        if commit:
            user.save()
        return user



