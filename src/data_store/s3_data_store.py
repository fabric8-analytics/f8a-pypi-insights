#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactions with Amazon S3.

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
import json
import logging
import os
import pickle

import boto3
import botocore
import daiquiri

from src.config.cloud_constants import AWS_S3_ENDPOINT_URL
from src.data_store.abstract_data_store import AbstractDataStore

daiquiri.setup(level=logging.ERROR)
_logger = daiquiri.getLogger(__name__)


class S3DataStore(AbstractDataStore):
    """S3 wrapper object."""

    def __init__(self, src_bucket_name, access_key, secret_key):
        """Create a new S3 notebooks store instance.

        :src_bucket_name: The name of S3 bucket to connect to
        :access_key: The access key for S3
        :secret_key: The secret key for S3

        :returns: An instance of the S3 notebooks store class
        """
        self.session = boto3.session.Session(aws_access_key_id=access_key,
                                             aws_secret_access_key=secret_key)
        if AWS_S3_ENDPOINT_URL == '':
            self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
                signature_version='s3v4'))
        else:
            self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
                signature_version='s3v4'), region_name='us-east-1',
                                                     endpoint_url=AWS_S3_ENDPOINT_URL)
        self.bucket = self.s3_resource.Bucket(src_bucket_name)
        self.bucket_name = src_bucket_name

    def get_name(self):
        """Get name of this object's bucket."""
        return "S3:" + self.bucket_name

    def read_json_file(self, filename):
        """Read JSON file from the S3 bucket."""
        return json.loads(self.read_generic_file(filename))

    def read_generic_file(self, filename):
        """Read a file from the S3 bucket."""
        obj = self.s3_resource.Object(self.bucket_name, filename).get()['Body'].read()
        utf_data = obj.decode("utf-8")
        return utf_data

    def read_pickle_file(self, filename):
        """Read pickle file from s3."""
        return pickle.loads(self.read_generic_file(filename), encoding='utf-8')

    def upload_file(self, src, target):
        """Upload file into notebooks store."""
        self.bucket.upload_file(src, target)
        return None

    def upload_folder_to_s3(self, folder_path, prefix=''):
        """Upload(Sync) a folder to S3.

        :folder_path: The local path of the folder to upload to s3
        :prefix: The prefix to attach to the folder path in the S3 bucket
        """
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if root != '.':
                    s3_dest = os.path.join(prefix, root, filename)
                else:
                    s3_dest = os.path.join(prefix, filename)
                self.bucket.upload_file(os.path.join(root, filename), s3_dest)
