from django import forms
from .models import (
    DataNeedsCorrection, 
    DataCorrected,
    DataValidated,
    WatCategory
)

class DataCorrectForm(forms.ModelForm):
    class Meta:
        model = DataCorrected
        fields = [
            'referenceData',
            'correctedText'
        ]


class DataValidateForm(forms.ModelForm):
    class Meta:
        model = DataValidated
        fields = [
            'referenceData',
        ]