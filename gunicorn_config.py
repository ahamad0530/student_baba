import os

# Gunicorn config for Render deployment
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
workers = 2
timeout = 120
preload_app = True
