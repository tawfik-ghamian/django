web: python manage.py migrate && gunicorn mysite.wsgi 
worker: celery -A mysite worker --loglevel=info 