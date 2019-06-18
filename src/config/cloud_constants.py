#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the constants for interaction with AWS.

Copyright © 2018 Red Hat Inc

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import numpy as np

from src.config.path_constants import DEPLOYMENT_PREFIX

USE_CLOUD_SERVICES = os.environ.get('USE_CLOUD_SERVICES', 'true') == 'true'
AWS_S3_ACCESS_KEY_ID = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')
AWS_S3_SECRET_KEY_ID = os.environ.get('AWS_S3_SECRET_ACCESS_KEY', '')
S3_BUCKET_NAME = DEPLOYMENT_PREFIX + '-' + os.environ.get('S3_BUCKET_NAME', 'hpf-pypi-insights')
AWS_S3_ENDPOINT_URL = os.environ.get('AWS_S3_ENDPOINT_URL', '')
MIN_CONFIDENCE_SCORE = np.float32(int(os.environ.get('MIN_CONFIDENCE_SCORE', 30)))
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
