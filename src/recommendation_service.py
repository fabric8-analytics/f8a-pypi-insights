#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the rest API for the recommender.

Copyright © 2018 Red Hat Inc.

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
import logging

import daiquiri
import flask
from flask import Flask, request

import src.config.cloud_constants as cloud_constants
from src.config.path_constants import ECOSYSTEM
from src.data_store.s3_data_store import S3DataStore
from src.models.predict_model import HPFScoring

app = Flask(__name__)

if cloud_constants.USE_CLOUD_SERVICES:
    s3_client = S3DataStore(src_bucket_name=cloud_constants.S3_BUCKET_NAME,
                            access_key=cloud_constants.AWS_S3_ACCESS_KEY_ID,
                            secret_key=cloud_constants.AWS_S3_SECRET_KEY_ID)

else:
    from src.data_store.local_filesystem import LocalFileSystem

    # Change the source directory here for local file system testing.
    s3_client = LocalFileSystem(src_dir='/Users/aagamshah/Documents/RedHat/')

recommender = HPFScoring(num_recommendations=10, data_store=s3_client)

daiquiri.setup(level=os.environ.get('FLASK_LOGGING_LEVEL', logging.INFO))
_logger = daiquiri.getLogger(__name__)


@app.route('/api/v1/liveness', methods=['GET'])
def liveness():
    """Define the liveness probe."""
    return flask.jsonify({}), 200


@app.route('/api/v1/readiness', methods=['GET'])
def readiness():
    """Define the readiness probe."""
    return flask.jsonify({"status": "ready"}), 200


@app.route('/api/v1/companion_recommendation', methods=['POST'])
def recommendation():
    """Endpoint to serve recommendations."""
    global recommender
    limit = 5
    response_json = []
    for recommendation_request in request.json:
        _logger.info("Input direct+transitive package list is......")
        input_packages = recommendation_request.get('package_list', []) +\
            recommendation_request.get("transitive_stack", [])
        _logger.info(input_packages)
        companions, missing = recommender.predict(
            input_stack=frozenset(recommendation_request['package_list'])
        )
        companions = [d for d in companions if d['package_name'] not in input_packages][:limit]
        response_json.append({
            "missing_packages": missing,
            "companion_packages": companions,
            "ecosystem": ECOSYSTEM
        })
        _logger.info("Sending response.....")
        _logger.info(response_json)
    return flask.jsonify(response_json), 200


if __name__ == '__main__':
    app.run(debug=True, port=6006)
