web: python manage.py migrate && gunicorn mysite.wsgi --bind 0.0.0.0:$PORT --timeout 300
worker: celery -A mysite worker --loglevel=info --pool=solo