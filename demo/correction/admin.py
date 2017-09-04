from django.contrib import admin

# Register your models here.
from .models import DataNeedsCorrection, DataCorrected, DataValidated, WatCategory

class DataNeedsCorrectionAdmin(admin.ModelAdmin):
    list_display = ('data_id', 'category', 'originalText', 'translatedText', 'status')
    class Meta:
        model = DataNeedsCorrection

class DataCorrectedAdmin(admin.ModelAdmin):
    class Meta:
        model = DataCorrected

class DataValidatedAdmin(admin.ModelAdmin):
    class Meta:
        model = DataValidated

class WatCategoryAdmin(admin.ModelAdmin):
    list_display = ('category_id', 'name', 'example')
    class Meta:
        model = WatCategory


admin.site.register(DataNeedsCorrection, DataNeedsCorrectionAdmin)
admin.site.register(DataCorrected, DataCorrectedAdmin)
admin.site.register(DataValidated, DataValidatedAdmin)
admin.site.register(WatCategory, WatCategoryAdmin)