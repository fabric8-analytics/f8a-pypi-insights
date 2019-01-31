"""Contains the path constants for S3 and local storage."""
import os

ECOSYSTEM = os.environ.get('HPF_SCORING_REGION', 'pypi')
MODEL_VERSION = os.environ.get('MODEL_VERSION', '2019-01-03')
DEPLOYMENT_PREFIX = os.environ.get('DEPLOYMENT_PREFIX', 'dev')
PACKAGE_TO_ID_MAP = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                 'trained-model/package-to-id-dict.json')
ID_TO_PACKAGE_MAP = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                 'trained-model/id-to-package-dict.json')
MANIFEST_TO_ID_MAP = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                  'trained-model/manifest-to-id.pickle')
HPF_MODEL_PATH = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                              'intermediate-model/HPF_model.pkl')
