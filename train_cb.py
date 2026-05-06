import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SplitBrainDetector.settings')
django.setup()

from app.model.catBoost import train_model
train_model()
print("Ready!")