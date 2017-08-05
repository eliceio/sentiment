from django.contrib import admin

# Register your models here.
from .models import Mainmenu

class MenuAdmin(admin.ModelAdmin):
    class Meta:
        model = Mainmenu

admin.site.register(Mainmenu, MenuAdmin)