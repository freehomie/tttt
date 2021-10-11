from django.db import models

# Create your models here.

# django.db.models.Model 클래스를 상속받음
'''
Question 테이블
   id integer primary key,
   question_text varchar(200),
   pub_date datetime
'''
class Question(models.Model) :
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    
    def __str__(self) :
        return self.question_text
    
'''
Choice 테이블
   id integer primary key,
   choice_text varchar(200),
   votes integer
   question_id integer => 외래키
'''
class Choice(models.Model) :
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes=models.IntegerField(default=0)
    def __str__(self) :
        return self.choice_text
    