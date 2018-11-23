#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the code that deals with the HPF piece for scoring.

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

import logging
import sys

import daiquiri
import pandas as pd

from src.config.path_constants import (HPF_MODEL_PATH, PACKAGE_TO_ID_MAP,
                                       ID_TO_PACKAGE_MAP, MANIFEST_TO_ID_MAP)

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


class HPFScoring:
    """This class contains logic for scoring the trained HPF model."""

    def __init__(self, num_recommendations, data_store):
        """Initialize HPFScoring instance.

        :param num_recommendations: number of recommendations to fetch from the model
        :param data_store: instance of s3_data_store or local_filesystem
        """
        self.m = num_recommendations
        self.s3_client = data_store
        self.recommender = self._load_model()
        self.package_to_id_map = self._load_package_to_id_map()
        self.id_to_package_map = self._load_id_to_package_map()
        self.manifest_to_id_map = self._load_manifest_to_id_map()

    def _load_model(self):
        """Load the model from s3."""
        return self.s3_client.read_pickle_file(HPF_MODEL_PATH)

    def _load_package_to_id_map(self):
        """Load package to id map."""
        return self.s3_client.read_json_file(PACKAGE_TO_ID_MAP)

    def _load_id_to_package_map(self):
        """Load id to package map."""
        return self.s3_client.read_json_file(ID_TO_PACKAGE_MAP)

    def _load_manifest_to_id_map(self):
        """Load manifest to id map."""
        return self.s3_client.read_pickle_file(MANIFEST_TO_ID_MAP)

    def _get_closest_manifest_file(self, input_stack):
        """Get closest manifest file.

        :param input_stack: Input stack of the user
        :return manifest_id, exact_match
        """
        manifest_id = self.manifest_to_id_map.get(input_stack)
        exact_match = True
        if not manifest_id:
            exact_match = False
            min_diff = sys.maxsize
            for idx, manifest in enumerate(self.manifest_to_id_map.keys()):
                diff = len(manifest.difference(input_stack))
                if input_stack.issubset(manifest) and diff < min_diff:
                    min_diff = diff
                    manifest_id = idx

        return manifest_id, exact_match

    def _map_input_to_package_ids(self, input_stack):
        """Map user input to package ids.

        :param input_stack: User's input stack
        :return: package_id_list, missing_packages
        """
        package_id_list = list()
        missing_packages = list()
        for package in input_stack:
            package_id = self.package_to_id_map.get(package)
            if package_id:
                package_id_list.append(package_id)
            else:
                missing_packages.append(package)

        return package_id_list, missing_packages

    def _get_packages_from_id(self, package_ids):
        """Get packages from their ids.

        :param package_ids: list of package ids
        :return: package_list
        """
        package_list = list()
        for i in package_ids:
            package = self.id_to_package_map.get(str(i))
            # We always have all packages recommended by model from the original package list.
            package_list.append(package)
        return package_list

    def predict(self, input_stack):
        """Predict companion packages for user stack

        :param input_stack: user stack
        :return: companion_packages, missing_packages
        """
        user_id, exact_match = self._get_closest_manifest_file(input_stack)
        package_id_list, missing_packages = self._map_input_to_package_ids(input_stack)
        companion_packages = list()
        if len(package_id_list) < len(missing_packages):
            _logger.info("Number of unknown packages more than known")
            return companion_packages, missing_packages
        if not (exact_match or user_id):
            _logger.info("Adding a new user....")

            try:
                counts_df = pd.DataFrame({
                    'ItemId': package_id_list,
                    'Count': [1] * len(package_id_list)
                })
                is_user_added = self.recommender.add_user(
                    user_id=self.recommender.nusers + 1,
                    counts_df=counts_df
                )
                if is_user_added:
                    recommendations = self.recommender.topN(
                        user=self.recommender.nusers - 1,
                        n=self.m
                    )
                else:
                    raise ValueError('Unable to add user')

            except ValueError as e:
                _logger.error(e)
                return companion_packages, missing_packages

        else:
            if exact_match:
                _logger.info("Found an exact match")
            else:
                _logger.info("Found a closest match")
            recommendations = self.recommender.topN(
                user=user_id,
                n=self.m
            )

        package_set = set(self._get_packages_from_id(recommendations))

        # Remove packages that were already seen by user.
        # TODO: Filter packages based on feedback as well.
        # TODO: Remove transitive dependencies as well.
        filtered_packages = package_set - input_stack

        # TODO: Calculate co-occurrence probability here.
        for package in filtered_packages:
            companion_packages.append({
                "package_name": package,
                "topic_list": []
            })

        return companion_packages, missing_packages
