from django import forms
# from .models import Video

class VideoUploadForm(forms.Form):
    # filename = forms.CharField(max_length=30)
    video_file = forms.FileField()
    # class Meta:
        # model = Video  