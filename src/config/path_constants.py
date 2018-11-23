"""Contains the path constants for S3 and local storage."""
import os

ECOSYSTEM = os.environ.get('HPF_SCORING_REGION', 'pypi')
PACKAGE_TO_ID_MAP = os.path.join(ECOSYSTEM, 'scoring/package-to-id-dict.json')
ID_TO_PACKAGE_MAP = os.path.join(ECOSYSTEM, 'scoring/id-to-package-dict.json')
MANIFEST_TO_ID_MAP = os.path.join(ECOSYSTEM, 'scoring/manifest-to-id.pickle')
HPF_MODEL_PATH = os.path.join(ECOSYSTEM, 'model/HPF_model.pkl')
