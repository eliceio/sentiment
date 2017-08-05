from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.views import View
from mainmenu.models import Mainmenu
from wat.views import IndexView


from .models import (
    DataNeedsCorrection,
    DataCorrected,
    DataValidated
)
from .forms import (
    DataCorrectForm, 
    DataValidateForm,
)

class CorrectView(IndexView):
    template = "correct.html"
    SIZE = 10
    form_class = DataCorrectForm
    context = dict(IndexView.context)
    context.update({
        "title": 'Data Correction',
        "form": form_class(),
        "ncs": DataNeedsCorrection.objects.random_defaults(SIZE)
    })
    def get_form(self):
        return self.form_class()

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)
    
    def form_valid(self, form):
        return HttpResponseRedirect('/success')
    
    def form_invalid(self, form):
        return HttpResponseRedirect('/failed')


class ValidateView(IndexView):
    template = "validate.html"
    form_class = DataValidateForm()
    context = dict(IndexView.context)
    context.update({
        "title":'Data Validation',
        "form":form_class,
        "cds": DataCorrected
    })
    def post(self, request, *args, **kwargs):
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)
    
