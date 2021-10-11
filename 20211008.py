# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:05:49 2021

@author: 32lhg
"""
############# Django 설치하기
'''
1. anaconda prompt
    - 장고설치
    pip install Django
    - 장고버전확인
    python -m django --version
    - 설정폴더로 위치 변경
    cd D:\R\Django>
    - 프로젝트생성
    django-admin startproject mysite
    - 프로젝트 폴더로 이동
      cd mysite
    - application 생성
    python manage.py startapp polls
2. spyder의 폴더위치를 D:\R\Django\mysite (설정폴더)로 변경
    - mysite 폴더
      settings.py 파일을 spyder에서 열기
3. anaconda prompt
   -db설정하기
    python manage.py migrate
   -관리자 생성하기
   python manage.py createsuperuser이미지
   - server 연결하기 (기본 8000번포트 설정됨)
   python manage.py runserver
   python manage.py runserver 0.0.0.0:9000 -> 9000번 포트를 열 수 있음
4. 브라우저 띄우기
    http://localhost:8000
    http://localhost:8000/admin => 로그인창
                        createsuperuser에 등록한 id로 로그인하기
5. 애플리케이션 개발하기 - spyder
    mysite/urls.py
    - path('polls/',views.index,name='index') #polls라는 어플리케이션 하나 추가
    
6. anaconda prompt
    - 폴더 이동
      cd D:\R\Django\mysite
    - db 설정
      python manage.py makemigrations => db변경내용 적용.
      python manage.py migrate
      python manage.py runserver -> 브라우저에 연결
      view -> 컨트롤러

FTPD -> 서버
   server에는 ip주소가 있음
   127.0.0.1 or localhost -> 나 => 컴퓨터 한 개
   컴퓨터 하나에 65500개 정도의 연결 가능한 방을 가지고 있음 => 포트
   1000번 이하 : 공용포트로 약속
   21 : ftp 포트
   22 : ssd
   25 : telnet
   80 : http
   장고 : 8000번을 기본으로 함(변경 가능)
   톰캣 : 8080
   오라클 : 1521
   localhost -> 컴퓨터, 8000 -> 포트 => localhost:8000 -> 컴퓨터의 8000번 포트로 들어감
   
'''
