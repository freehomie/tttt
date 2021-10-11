#polls/views.py
from django.shortcuts import render,get_object_or_404
from django.http import HttpResponse,HttpResponseRedirect
from polls.models import Question,Choice
from django.urls import reverse

# Create your views here.
# http://localhost:8000/polls/
#def index(request):
#    return HttpResponse("Hello, world. You're at the polls index.")
def index(request):
    #Question.objects.all() : question 테이블의 모든 정보 조회
    latest_question_list = Question.objects.all().order_by('-pub_date')[:5]
    context = {'latest_question_list': latest_question_list}
    #template 설정 : 'polls/index.html'
    return render(request, 'polls/index.html', context)    

#http://localhost:8000/polls/1/
def detail(request,question_id):
#get_object_or_404 : question_id 값이 기본키인 데이터를 조회.
#                    해당 내용이 없으면 404오류 발생.  
#question : question_id의 데이터 정보 
    question = get_object_or_404(Question,pk=question_id)
    return render(request,'polls/detail.html',{'question':question})

#http://localhost:8000/polls/1/vote/
def vote(request,question_id):
    question = get_object_or_404(Question,pk=question_id)
    try :
      #request.POST['choice'] : choice파라미터의 값
      #selected_choice : 선택된 Choice값에 해당하는 Choice 객체
     selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        return render(request,'polls/detail.html',{
            'question':question,
            'error_message':"Your didn't select a choice"
            })
    else :
        selected_choice.votes += 1
        selected_choice.save() #db에 수정
        # polls/1/result
        return HttpResponseRedirect\
            (reverse('polls:results',args=(question.id,)))
    
#http://localhost:8000/polls/1/results/
def results(request,question_id) :
    question = get_object_or_404(Question,pk=question_id)
    return render(request,'polls/results.html',{'question':question})
    
    
