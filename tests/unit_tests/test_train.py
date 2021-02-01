#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains tests for training code for pypi insights.

Copyright © 2019 Red Hat Inc

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
import pickle
import mock
import pandas as pd
from fractions import Fraction
from training.train import preprocess_raw_data, make_user_item_df, format_dict, train_test_split

with open('tests/data/manifest.json', 'r') as f:
    manifest = json.load(f)
with open('tests/data/manifest-to-id.pickle', 'rb') as f:
    manifest_to_id_dict = pickle.load(f)
with open('tests/data/package-to-id-dict.json', 'r') as f:
    package_to_id_dict = json.load(f)
with open('tests/data/user-item-list.json', 'r') as f:
    user_item_list = json.load(f)


def mock_validate_manifest_data(x):
    """Mock validate manifest data."""
    return x


class TestTraining:
    """Test the training part."""

    @mock.patch("training.train.validate_manifest_data", side_effect=mock_validate_manifest_data)
    def test_preprocess_raw_data(self, _):
        """Test preprocessing of raw data."""
        package_id_dict, id_package_dict, manifest_id_dict = preprocess_raw_data(
            raw_data_dict=manifest.get('package_dict'),
            lower_limit=1,
            upper_limit=100)
        assert package_id_dict == package_to_id_dict
        for man in manifest_id_dict:
            assert man in manifest_to_id_dict

    def test_make_user_item_df(self):
        """Test make user item data frame."""
        user_item_df = make_user_item_df(manifest_dict=manifest_to_id_dict,
                                         package_dict=package_to_id_dict,
                                         user_input_stacks=manifest.get('package_dict')
                                         .get('user_input_stack'))

        for user_item in user_item_df:
            assert user_item in user_item_list

    def test_format_dict(self):
        """Test format dict."""
        format_pkg_id_dict, format_mnf_id_dict = format_dict(
            package_id_dict=package_to_id_dict,
            manifest_id_dict=manifest_to_id_dict)

        assert format_pkg_id_dict == {
            "ecosystem": "pypi",
            'package_list': package_to_id_dict
        }

        assert format_mnf_id_dict == {
            "ecosystem": "pypi",
            "manifest_list": manifest_to_id_dict
        }

    def test_train_test_split(self):
        """Test train test split."""
        user_item_df = pd.DataFrame(user_item_list)
        user_input_df = user_item_df.loc[user_item_df['is_user_input_stack']]
        train_df, test_df = train_test_split(user_item_df)
        assert round(float(Fraction(len(test_df.index),
                                    len(user_input_df.index))), 2) == 0.20
