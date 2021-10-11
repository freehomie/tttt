# -*- coding: utf-8 -*-
# polls/urls.py
from django.urls import path
from polls import views
app_name = 'polls'
urlpatterns = [
    path('',views.index,name='index'),  # /polls/
    path('<int:question_id>/',views.detail,name='detail'),  # /polls/1/
    path('<int:question_id>/vote/',views.vote,name='vote'),  # /polls/1/vote/
    path('<int:question_id>/results/', views.results, name='results'),   # /polls/5/results/
]