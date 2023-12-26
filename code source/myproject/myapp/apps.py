import os
import joblib
from django.apps import AppConfig
from django.conf import settings


class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'
    MODEL_FILE = os.path.join(settings.MODELS, "mlp_classifier.joblib")
    model = joblib.load(MODEL_FILE)
