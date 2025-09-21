import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

app = Celery('tasks',
             backend=os.environ.get('REDIS_URL'),
             broker=os.environ.get('REDIS_URL')
             )

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Test connection on startup
try:
    # Test if broker is accessible
    app.control.inspect().ping()
    print("Celery broker connection successful")
except Exception as e:
    print(f"Warning: Celery broker connection failed: {e}")
    print("Falling back to synchronous processing...")

@app.task(bind=True,soft_time_limit=900)
def debug_task(self):
    print(f'Request: {self.request!r}')