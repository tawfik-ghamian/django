web: python manage.py migrate && gunicorn mysite.wsgi --bind 0.0.0.0:$PORT --timeout 600 --workers 1
worker: celery -A mysite worker --loglevel=info --pool=solo