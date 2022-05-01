from django.db import models
from django.shortcuts import render
from django.conf import settings
from reflections.models import reflection
from modelcluster.fields import ParentalKey
from django.core.files.storage import default_storage
import cv2
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
import numpy as np
from streams import blocks

from scipy import stats
from pathlib import Path
import sqlite3, datetime, os, uuid, glob

# Create your models here.
from cam_app import predictModel


class PlayVideoPage(Page):
    """Play Video Page."""

    template = "play_video/play_video.html"

    max_count = 2
    reflectionlists={}
    length = float(0)
    name_title = models.CharField(max_length=100, blank=False, null=True)
    video_url = models.URLField(max_length=100, blank=False, null=True)
    video_num = models.PositiveIntegerField(blank = False)
    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("video_url"),
                FieldPanel("video_num"),
            ],
            heading="Page Options",
        ),
    ]


    def serve(self, request):
        reflectionList = list(reflection.objects.filter(video_num=self.video_num))
        reflectionList_numpy = np.matrix(reflectionList[0].video_reflection)
        # list_reflections=list()
        reflectionList.pop(0)
        for reflection_item in reflectionList:
            numpy_result = np.append(reflectionList_numpy,np.matrix(reflection_item.video_reflection),axis=0)
        # print("# a的每⼀列中最常见的成员为：{}，分别出现了{}次。".format(stats.mode(numpy_result)[0][0], stats.mode(numpy_result)[1][0]))
        self.reflectionlists={}
        self.reflectionlists=stats.mode(numpy_result)[0][0]
        self.length=1/len(self.reflectionlists)
        #     for i in np.hsplit(numpy_result, np.shape(numpy_result)[1]):
        #         print(np.shape(i.transpose()))
        #         print(i.transpose())
        #         list_reflections.append(np.argmax(np.bincount(i.transpose())))
        # print(numpy_result)
        # print(list_reflections)
        if (request.FILES):
            for file_obj in request.FILES.getlist("file_data"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"media/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                reflectionList=[]
                vc = cv2.VideoCapture("mysite/media/media/"+filename)
                # print("mysite/media/media/"+filename)
                if predictModel.model == None:
                    predictModel.load_model()
                rval, frame = vc.read()
                # frame_count = 1
                # count = 0vc.read
                frame_count = 0
                fps = vc.get(cv2.CAP_PROP_FPS)
                # print(int(fps))
                while rval:
                    rval, frame = vc.read()
                    # videotime = vc.get(cv2.CAP_PROP_POS_MSEC)
                    frame_count=frame_count+1
                    if int(frame_count) % int(fps)  == 0:
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))
                        # print(faces)
                        if len(faces) != 0:
                            for (x, y, w, h) in faces:
                                face = gray[y:y + h, x:x + w]
                                face = cv2.resize(face, (48, 48))
                                face_arr = face.astype(np.float32)
                                face_arr /= 255.
                                face_arr = np.expand_dims(face_arr, axis=0)
                                predictions = predictModel.model.predict(face_arr)
                                i = np.argmax(predictions, axis=1)
                        else: i=8
                        reflectionList.append(i)
                results = np.array(reflectionList)
                n = reflection(video_num=self.video_num,video_reflection=results.transpose())
                n.save()
            return render(request, "play_video/play_video.html", {'page': self})

        return render(request, "play_video/play_video.html", {'page': self})
