#!/bin/bash

gunicorn --pythonpath /opt/app-root/src/src -b 0.0.0.0:$SERVICE_PORT --workers=2 -k sync -t $SERVICE_TIMEOUT recommendation_service:app --log-level $FLASK_LOGGING_LEVEL
