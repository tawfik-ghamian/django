web: gunicorn mysite.wsgi:application 
worker: celery -A mysite worker --loglevel=info 