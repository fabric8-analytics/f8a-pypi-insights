#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the constants for interaction with path.

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

ECOSYSTEM = os.environ.get('HPF_SCORING_REGION', 'pypi')
MODEL_VERSION = os.environ.get('MODEL_VERSION', '2019-01-03')
PACKAGE_TO_ID_MAP = os.path.join(MODEL_VERSION,
                                 'trained-model/package-to-id-dict-without-trans.json')
ID_TO_PACKAGE_MAP = os.path.join(MODEL_VERSION,
                                 'trained-model/id-to-package-dict-without-trans.json')
MANIFEST_TO_ID_MAP = os.path.join(MODEL_VERSION,
                                  'trained-model/manifest-to-id-without-trans.pickle')
HPF_MODEL_PATH = os.path.join(MODEL_VERSION,
                              'intermediate-model/Bayesian_Sets.pkl')
MANIFEST_PATH = os.path.join(MODEL_VERSION,
                             'data/manifest.json')
HYPERPARAMETERS_PATH = os.path.join(MODEL_VERSION,
                                    'intermediate-model/hyperparameters.json')
