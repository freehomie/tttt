from django.contrib import admin

#Admin사이트에 Question,Choice테이블 반영하기
# Register your models here.

from polls.models import Question,Choice
# Question과 Choice를 하나의 화면에서 입력받도록 설정하기
class ChoiceInline(admin.StackedInline):
    model = Choice
    extra = 2  #초기값
    
class QuestionAdmin(admin.ModelAdmin) :
    #admin 추가시 컬럼의 위치 설정.
#    fields = ['pub_date','question_text']
    fieldsets = [
        ('Question Statement', {'fields': ['question_text']}),
        ('Date Information', {'fields': ['pub_date'],'classes': ['collapse']}),
    ]
    inlines = [ChoiceInline]

admin.site.register(Question,QuestionAdmin)
admin.site.register(Choice)
