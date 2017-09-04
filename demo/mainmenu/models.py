from django.db import models

# Create your models here.
class Mainmenu(models.Model):
    name = models.CharField(max_length=30,)
    description = models.TextField()
    slug = models.SlugField(max_length=50, db_index=True)
    path = models.CharField(max_length=30,)
    order = models.PositiveIntegerField()
    isVisible = models.BooleanField(default=False)

    class Meta:
        ordering = ["order"]


    def __unicode__(self):
        return self.name
    
    def __str__(self):
        return self.name

