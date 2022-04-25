from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_sameorigin
import uuid
from django.views import View
# Create your views here.
from django.http import HttpResponse
str_uuid = uuid.uuid4()  # The UUID for image uploading

class PlayVideoView(View):
    @xframe_options_sameorigin
    def get(self, request):
        return render(request, 'play_video/play_video.html')