# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2017-07-22 13:32
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('correction', '0008_auto_20170722_2122'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='dataneedscorrection',
            options={'verbose_name_plural': 'Data needs to be corrected'},
        ),
    ]