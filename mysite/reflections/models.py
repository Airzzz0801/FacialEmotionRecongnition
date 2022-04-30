from django.db import models

class reflection(models.Model):
    video_num = models.PositiveIntegerField(blank = False)
    video_reflection = models.TextField()
