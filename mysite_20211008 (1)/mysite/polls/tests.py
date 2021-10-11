from django.test import TestCase
from .models import Question
# Create your tests here.
#latest_question_list = Question.objects.all()[:5]
#latest_question_list

class QuestionModelTests(TestCase):
    def test_was_published_recently_with_future_question(self):
         latest_question_list = Question.objects.all()[:5]
         print(latest_question_list)
