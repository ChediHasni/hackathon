"""
Django settings for zonescorer project.
"""

from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='django-insecure-zonescorer-dev-key-change-in-production-xyz')

def _config_bool(name, default=False):
    raw = config(name, default=None)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    return default


DEBUG = _config_bool('DEBUG', default=True)

ALLOWED_HOSTS = [
    host.strip()
    for host in config('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')
    if host.strip()
]

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'zonescorer.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR.parent / 'frontend' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'zonescorer.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
}

CORS_ALLOW_ALL_ORIGINS = True

# ─── GeoAI Credentials (read from .env) ───────────────────────────────────────
GEE_SERVICE_ACCOUNT = config('GEE_SERVICE_ACCOUNT', default='')
GEE_KEY_FILE = config('GEE_KEY_FILE', default='')
CDS_API_KEY = config('CDS_API_KEY', default='')
TRANSITLAND_API_KEY = config('TRANSITLAND_API_KEY', default='')

# ─── GNN Model Path ────────────────────────────────────────────────────────────
GNN_MODEL_PATH = BASE_DIR / 'gnn' / 'model.pt'
