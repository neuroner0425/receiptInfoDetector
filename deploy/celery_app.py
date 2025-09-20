from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0', include=['tasks'])
app.config_from_object('celeryconfig')

# pip install celery
# pip install redis
# brew install redis
# brew services start redis
# sudo celery multi start 6 -A celery_app --concurrency=8 --logfile=shell.log --pool=gevent 
# sudo celery multi stop 6