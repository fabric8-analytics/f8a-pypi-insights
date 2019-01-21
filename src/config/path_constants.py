"""Contains the path constants for S3 and local storage."""
import os

ECOSYSTEM = os.environ.get('HPF_SCORING_REGION', 'pypi')
MODEL_VERSION = os.environ.get('MODEL_VERSION', '2019-01-03')
PACKAGE_TO_ID_MAP = os.path.join(ECOSYSTEM, MODEL_VERSION, 'scoring/package-to-id-dict.json')
ID_TO_PACKAGE_MAP = os.path.join(ECOSYSTEM, MODEL_VERSION, 'scoring/id-to-package-dict.json')
MANIFEST_TO_ID_MAP = os.path.join(ECOSYSTEM, MODEL_VERSION, 'scoring/manifest-to-id.pickle')
HPF_MODEL_PATH = os.path.join(ECOSYSTEM, MODEL_VERSION, 'model/HPF_model.pkl')
