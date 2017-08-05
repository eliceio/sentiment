from django.db import models
from django.utils import timezone
import random as rd
# Create your models here.



class DNCQuerySet(models.QuerySet):
    def defaults(self):
        return self.filter(status='default')

    def corrections(self):
        return self.filter(status='corrected')

    def validations(self):
        return self.filter(status='validated')

    def random_defaults(self, size):
        rd.seed(rd.randint(1, 10000))
        qs = self.filter(status='default') 
        return [qs.filter(data_id=rd.randint(1, len(qs)))[0] for _ in range(size)]

class DataNeedsCorrection(models.Model):
    data_id = models.AutoField(primary_key=True)
    category = models.ForeignKey('WatCategory', on_delete=models.CASCADE)
    originalText = models.CharField(max_length=300)
    translatedText = models.CharField(max_length=300)
    status = models.CharField(max_length=20, choices=(("default", "Default"),("corrected","Corrected"),("validated", "Validated")))
    created = models.DateTimeField(editable=False)
    modified = models.DateTimeField()
    objects = DNCQuerySet.as_manager()

    class Meta:
        verbose_name_plural = "Data needs to be corrected"

    def __unicode__(self):
        return self.originalText

    def __str__(self):
        return self.originalText

    def save(self, *args, **kwargs):
        if not self.data_id:
            self.created = timezone.now()
        self.modified = timezone.now()
        return super(DataNeedsCorrection, self).save(*args, **kwargs)




class DataCorrected(models.Model):
    data_id = models.AutoField(primary_key=True)
    referenceData = models.ForeignKey('DataNeedsCorrection', on_delete=models.CASCADE)
    correctedText = models.CharField(max_length=300)
    created = models.DateTimeField(editable=False)
    modified = models.DateTimeField()
    class Meta:
        verbose_name_plural = 'Data Corrected'

    def __unicode__(self):
        return self.correctedText

    def __str__(self):
        return self.correctedText

    def save(self, *args, **kwargs):
        if not self.data_id:
            self.created = timezone.now()
        self.modified = timezone.now()
        return super(DataCorrected, self).save(*args, **kwargs)

class DataValidated(models.Model):
    data_id = models.AutoField(primary_key=True)
    referenceData = models.ForeignKey('DataNeedsCorrection', on_delete=models.CASCADE)
    created = models.DateTimeField(editable=False)
    modified = models.DateTimeField()

    class Meta:
        verbose_name_plural = "Validated Data"

    def save(self, *args, **kwargs):
        if not self.data_id:
            self.created = timezone.now()
        self.modified = timezone.now()
        return super(DataValidated,self).save(*args, **kwargs)

class WatCategory(models.Model):
    category_id = models.PositiveIntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    example = models.CharField(max_length=300)

    class Meta:
        verbose_name_plural = "Word Analogy Reasoning Task Categories"

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name

