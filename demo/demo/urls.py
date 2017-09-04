from django.conf.urls import url
from . import apis
from .views import DemoView
urlpatterns = [
    url(r'^$', DemoView.as_view(), name="demo"),
    url(r'^svm/(?P<C>\d+\.\d+)/$', apis.svm),
    url(r'^clf/(?P<sentence>[\w|\W]+)/$', apis.clf),
    url(r'^wat/(?P<sentence>[\w|\W]+)/$', apis.wat),
]