from django.shortcuts import render
from wat.views import IndexView


class DemoView(IndexView):
    template = "demo.html"
    context = dict(IndexView.context)





