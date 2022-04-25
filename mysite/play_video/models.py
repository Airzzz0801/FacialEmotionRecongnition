from django.db import models
from django.shortcuts import render
from django.conf import settings

from modelcluster.fields import ParentalKey

from wagtail.admin.edit_handlers import (
    FieldPanel,
    MultiFieldPanel,
    InlinePanel,
    StreamFieldPanel,
    PageChooserPanel,
)
from wagtail.core.models import Page, Orderable
from wagtail.core.fields import RichTextField, StreamField
from wagtail.images.edit_handlers import ImageChooserPanel

from streams import blocks

import sqlite3, datetime, os


# Create your models here.
class PlayVideoPage(Page):
    """Play Video Page."""

    template = "play_video/play_video.html"

    max_count = 2

    name_title = models.CharField(max_length=100, blank=True, null=True)
    video_url = models.CharField(max_length=100, blank=True, null=True)
    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("video_url"),

            ],
            heading="Page Options",
        ),
    ]

    def serve(self, request):
        return render(request, "play_video/play_video.html", {'page': self})
