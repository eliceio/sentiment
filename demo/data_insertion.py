# -*- coding:utf-8 -*-
from correction.models import DataNeedsCorrection, WatCategory
from django.utils import timezone
categories_dict = {category.name:category for category in WatCategory.objects.all()}
current_category = -1
line_limit = 8875
# line_limit = 10
with open('./correction/assets/checked.txt', encoding='utf-8') as fa, \
    open('./correction/assets/wat.txt', encoding='utf-8') as fb, \
    open('./correction/assets/faied.txt', encoding='utf-8', mode='w+') as fc:
    for i in range(line_limit):
        translatedText, originalText = fa.readline().strip(), fb.readline().strip()
        # print(translatedText, originalText)
        if not originalText or len(originalText) <= 1:
            continue
        if originalText[0] == ':':
            _, cc = originalText.split(":")
            current_category = categories_dict[cc.strip()]
            continue
        try:
            DataNeedsCorrection.objects.create(
                data_id = i,
                category = current_category,
                originalText = originalText,
                translatedText = ' '.join(translatedText.split()[1:]),
                modified = timezone.now(),
                created = timezone.now(),
                status = "default"
            )
        except:
            fc.write(str(i) + originalText + "\n")


