from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.views import View
from mainmenu.models import Mainmenu


class IndexView(View):
    template = "index.html"
    context = {
        "menus": list(filter(lambda x:x.isVisible, Mainmenu.objects.all())),
        "title": "Korean Sentiment Analysis"
    }
    def get(self, req, *args, **kwargs):
        return render(req, self.template, self.context)


class SuccessView(IndexView):
    template = "success.html"

class FailedView(IndexView):
    template = "failed.html"

class WatView(IndexView):
    template = 'wat.html'
    