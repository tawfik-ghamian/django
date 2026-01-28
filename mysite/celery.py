# import os

# from celery import Celery

# # Set the default Django settings module for the 'celery' program.
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

# # Get Redis URL from environment
# redis_url = os.environ.get('REDIS_URL')

# if not redis_url:
#     print("=" * 80)
#     print("❌ CRITICAL ERROR: REDIS_URL not found!")
#     print("=" * 80)
#     raise EnvironmentError("REDIS_URL environment variable is required but not set")

# print(f"🔗 Connecting to Redis: {redis_url[:50]}...")

# app = Celery(
#     'mysite',
#     broker=redis_url,
#     backend=redis_url
# )

# # Using a string here means the worker doesn't have to serialize
# # the configuration object to child processes.
# # - namespace='CELERY' means all celery-related configuration keys
# #   should have a `CELERY_` prefix.
# app.config_from_object('django.conf:settings', namespace='CELERY')

# # Load task modules from all registered Django apps.
# app.autodiscover_tasks()

# # Test connection on startup
# try:
#     # Test if broker is accessible
#     app.control.inspect().ping()
#     print("Celery broker connection successful")
# except Exception as e:
#     print(f"Warning: Celery broker connection failed: {e}")
#     print("Falling back to synchronous processing...")

# @app.task(bind=True)
# def debug_task(self):
#     print(f'Request: {self.request!r}')


import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

# Get Redis URL from environment
redis_url = os.environ.get('REDIS_URL',"http://localhost:8000")

if not redis_url:
    print("=" * 80)
    print("❌ CRITICAL ERROR: REDIS_URL not found!")
    print("=" * 80)
    raise EnvironmentError("REDIS_URL environment variable is required but not set")

print(f"🔗 Connecting to Redis: {redis_url[:50]}...")

# Create Celery app
app = Celery('mysite')

# Configure Celery using settings from Django
app.config_from_object('django.conf:settings', namespace='CELERY')

# Manually set broker and backend (ensures it uses Redis)
app.conf.broker_url = redis_url
app.conf.result_backend = redis_url

# Additional broker settings for reliability
app.conf.broker_connection_retry_on_startup = True
app.conf.broker_connection_retry = True
app.conf.broker_connection_max_retries = 10

# Load task modules from all registered Django apps
app.autodiscover_tasks()

print("✅ Celery app configured successfully")

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')