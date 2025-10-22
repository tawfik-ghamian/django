import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

# Get Redis URL from environment
redis_url = os.environ.get('REDIS_URL')

if not redis_url:
    print("❌ ERROR: REDIS_URL environment variable is not set!")
    redis_url = 'redis://localhost:6379/0'  # Fallback
    print(f"⚠️ Falling back to: {redis_url}")

print(f"🔗 Connecting to Redis: {redis_url[:50]}...")

app = Celery('tasks',
             backend=redis_url,
             broker=redis_url
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

@app.task(bind=True,soft_time_limit=1200)
def debug_task(self):
    print(f'Request: {self.request!r}')