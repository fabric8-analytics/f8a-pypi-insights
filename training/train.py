#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains training code for pypi insights.

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

from rudra.data_store.aws import AmazonS3
from rudra.utils.helper import load_hyper_params
from rudra.utils.validation import BQValidation
from fractions import Fraction
import pandas as pd
import numpy as np
import hpfrec
import json
import logging
import subprocess
import yaml

from src.config.path_constants import (PACKAGE_TO_ID_MAP, ID_TO_PACKAGE_MAP,
                                       MANIFEST_TO_ID_MAP, MANIFEST_PATH, HPF_MODEL_PATH, ECOSYSTEM,
                                       HYPERPARAMETERS_PATH, MODEL_VERSION, DEPLOYMENT_PREFIX)
from src.config.cloud_constants import (AWS_S3_BUCKET_NAME,
                                        AWS_S3_SECRET_KEY_ID, AWS_S3_ACCESS_KEY_ID, GITHUB_TOKEN)

logging.basicConfig()
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

bq_validator = BQValidation()

DEPLOYMENT_PREFIX_ENVIRONMENT_NAME_MAP = {
    'stage': 'staging',
    'prod': 'production'
}


def load_s3():  # pragma: no cover
    """Create connection s3."""
    s3_object = AmazonS3(bucket_name=AWS_S3_BUCKET_NAME,
                         aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_S3_SECRET_KEY_ID)

    s3_object.connect()
    if s3_object.is_connected():
        _logger.info("S3 connection established for {} bucket".format(AWS_S3_BUCKET_NAME))
        return s3_object

    raise Exception("S3 Connection Failed")


def load_data(s3_client):  # pragma: no cover
    """Load data from s3 bucket."""
    _logger.info("Reading Manifest file from {} path".format(MANIFEST_PATH))
    raw_data_dict = s3_client.read_json_file(MANIFEST_PATH)
    if raw_data_dict is None:
        raise Exception("manifest.json not found")
    _logger.info("Size of Raw Manifest file is: {}".format(len(raw_data_dict)))
    return raw_data_dict


def make_user_item_df(manifest_dict, package_dict, user_input_stacks):
    """Make user item dataframe."""
    user_item_list = []
    set_input_stacks = {frozenset(x) for x in user_input_stacks}
    for manifest, user_id in manifest_dict.items():
        is_user_input_stack = manifest in set_input_stacks
        for package in manifest:
            item_id = package_dict[package]
            user_item_list.append(
                {
                    "UserId": user_id,
                    "ItemId": item_id,
                    "Count": 1,
                    "is_user_input_stack": is_user_input_stack
                }
            )
    return user_item_list


def generate_package_id_dict(manifest_list):
    """Generate package id dictionary."""
    package_id_dict = dict()
    id_package_dict = dict()
    count = 0
    for manifest in manifest_list:
        for package_name in manifest:
            if package_name not in package_id_dict:
                package_id_dict[package_name] = count
                id_package_dict[count] = package_name
                count += 1
    return package_id_dict, id_package_dict


def format_dict(package_id_dict, manifest_id_dict):
    """Format the dictionaries."""
    format_pkg_id_dict = {'ecosystem': ECOSYSTEM,
                          'package_list': package_id_dict
                          }
    format_mnf_id_dict = {'ecosystem': ECOSYSTEM,
                          'manifest_list': manifest_id_dict
                          }
    return format_pkg_id_dict, format_mnf_id_dict


def generate_manifest_id_dict(manifest_list):
    """Generate manifest id dictionary."""
    count = 0
    manifest_id_dict = dict()
    manifest_set = {frozenset(x) for x in manifest_list}
    _logger.info("Number of unique manifests are: {}".format(len(manifest_set)))
    for manifest in manifest_set:
        manifest_id_dict[manifest] = count
        count += 1
    return manifest_id_dict


def run_recommender(train_df, latent_factors):  # pragma: no cover
    """Start the recommender."""
    recommender = hpfrec.HPF(k=latent_factors, random_seed=123,
                             ncores=-1, stop_crit='train-llk', verbose=True,
                             reindex=False, stop_thr=0.000001, maxiter=3000)
    recommender.step_size = None
    _logger.warning("Model is training, Don't interrupt.")
    recommender.fit(train_df)
    return recommender


def validate_manifest_data(manifest_list):  # pragma: no cover
    """Validate manifest packages with pypi."""
    for idx, manifest in enumerate(manifest_list):
        filtered_manifest = bq_validator.validate_pypi(manifest)
        # Even the filtered manifest is of length 0, we don't care about that here.
        manifest_list[idx] = filtered_manifest


def preprocess_raw_data(raw_data_dict, lower_limit, upper_limit):
    """Preprocess raw data."""
    all_manifest_list = raw_data_dict.get('user_input_stack', []) \
        + raw_data_dict.get('bigquery_data', [])
    _logger.info("Number of manifests collected = {}".format(
        len(all_manifest_list)))
    validate_manifest_data(all_manifest_list)
    _logger.info("Manifest list now contains only packages from pypi")
    trimmed_manifest_list = [
        manifest for manifest in all_manifest_list if lower_limit < len(manifest) < upper_limit]
    _logger.info("Number of trimmed manifest = {}".format(
        len(trimmed_manifest_list)))
    package_id_dict, id_package_dict = generate_package_id_dict(trimmed_manifest_list)
    manifest_id_dict = generate_manifest_id_dict(trimmed_manifest_list)
    return package_id_dict, id_package_dict, manifest_id_dict


# Calculating DataFrame according to fraction
def extra_df(frac, data_df, train_df):
    """Calculate extra dataframe."""
    remain_frac = float("%.2f" % (0.80 - frac))
    len_df = len(data_df.index)
    no_rows = round(remain_frac * len_df)
    df_remain = pd.concat([data_df, train_df]).drop_duplicates(keep=False)
    df_remain_rand = df_remain.sample(frac=1)
    return df_remain_rand[:no_rows]


def train_test_split(data_df):
    """Split for training and testing."""
    user_input_df = data_df.loc[data_df['is_user_input_stack']]
    user_input_df = user_input_df.sample(frac=1)
    df_user = user_input_df.drop_duplicates(['UserId'])
    user_input_df = user_input_df.sample(frac=1)
    df_item = user_input_df.drop_duplicates(['ItemId'])
    train_df = pd.concat([df_user, df_item]).drop_duplicates()
    fraction = round(float(Fraction(len(train_df.index),
                                    len(user_input_df.index))), 2)

    if fraction < 0.80:
        df_ = extra_df(fraction, user_input_df, train_df)
        train_df = pd.concat([train_df, df_])
    test_df = pd.concat([user_input_df, train_df]).drop_duplicates(keep=False)
    test_df = test_df.drop(columns=['is_user_input_stack'])
    data_df = data_df.loc[~data_df['is_user_input_stack']]
    train_df = pd.concat([data_df, train_df])
    train_df = train_df.drop(columns=['is_user_input_stack'])
    _logger.info("Size of Training DF {} and Testing DF are: {}".format(
        len(train_df), len(test_df)))
    return train_df, test_df


# Calculating recall according to no of recommendations
def recall_at_m(m, test_df, recommender, user_count):
    """Calculate recall at `m`."""
    recall = []
    for i in range(user_count):
        x = np.array(test_df.loc[test_df.UserId.isin([i])].ItemId)
        rec_l = x.size
        recommendations = recommender.topN(user=i, n=m, exclude_seen=True)
        intersection_length = np.intersect1d(x, recommendations).size
        try:
            recall.append(intersection_length / rec_l)
        except ZeroDivisionError:
            pass
    return np.mean(recall)


def precision_at_m(m, test_df, recommender, user_count):
    """Calculate precision at `m`."""
    precision = []
    for i in range(user_count):
        x = np.array(test_df.loc[test_df.UserId.isin([i])].ItemId)
        recommendations = recommender.topN(user=i, n=m, exclude_seen=True)
        r_size = recommendations.size
        intersection_length = np.intersect1d(x, recommendations).size
        try:
            precision.append(intersection_length / r_size)
        except ZeroDivisionError:
            pass
    return np.mean(precision)


def precision_recall_at_m(m, test_df, recommender, user_item_df):
    """Precision and recall at given `m`."""
    user_count = user_item_df['UserId'].nunique()
    try:
        precision = precision_at_m(m, test_df, recommender, user_count)
        recall = recall_at_m(m, test_df, recommender, user_count)
        _logger.info("Precision {} and Recall are: {}".format(
            precision, recall))
        return precision, recall
    except ValueError:
        pass


def save_model(s3_client, recommender):  # pragma: no cover
    """Save model on s3."""
    try:
        status = s3_client.write_pickle_file(HPF_MODEL_PATH, recommender)
        _logger.info("Model has been saved {}.".format(status))
    except Exception as exc:
        _logger.error(str(exc))


def save_dictionaries(s3_client, package_id_dict,
                      id_package_dict, manifest_id_dict):  # pragma: no cover
    """Save the dictionaries for scoring."""
    pkg_status = s3_client.write_json_file(PACKAGE_TO_ID_MAP,
                                           package_id_dict)
    id_status = s3_client.write_json_file(ID_TO_PACKAGE_MAP,
                                          id_package_dict)
    mnf_status = s3_client.write_pickle_file(MANIFEST_TO_ID_MAP,
                                             manifest_id_dict)

    if not (pkg_status and mnf_status and id_status):
        raise Exception("Unable to store data files for scoring")

    logging.info("Saved dictionaries successfully")


def save_hyperparams(s3_client, content_json):
    """Save hyperparameters."""
    status = s3_client.write_json_file(HYPERPARAMETERS_PATH, content_json)
    if not status:
        raise Exception("Unable to store hyperparameters file")
    _logger.info("Hyperparameters saved")


def save_obj(s3_client, trained_recommender, hyper_params,
             package_id_dict, id_package_dict, manifest_id_dict):
    """Save the objects in s3 bucket."""
    _logger.info("Trying to save the model.")
    save_model(s3_client, trained_recommender)
    save_dictionaries(s3_client, package_id_dict, id_package_dict, manifest_id_dict)
    save_hyperparams(s3_client, hyper_params)


def build_hyperparams(lower_limit, upper_limit, latent_factor,
                      precision_30, recall_30, precision_50, recall_50):
    """Build hyper parameter object."""
    return {
        "deployment": DEPLOYMENT_PREFIX_ENVIRONMENT_NAME_MAP.get(DEPLOYMENT_PREFIX, ''),
        "model_version": MODEL_VERSION,
        "minimum_length_of_manifest": lower_limit,
        "maximum_length_of_manifest": upper_limit,
        "latent_factor": latent_factor,
        "precision_at_30": precision_30,
        "recall_at_30": recall_30,
        "f1_score_at_30": 2 * ((precision_30 * recall_30) / (precision_30 + recall_30)),
        "precision_at_50": precision_50,
        "recall_at_50": recall_50,
        "f1_score_at_50": 2 * ((precision_50 * recall_50) / (precision_50 + recall_50)),
    }


def exec_command(command_args, max_wait_time):
    """Execute the given command with arguments and perform error checks."""
    try:
        t = subprocess.Popen(command_args, shell=False)
        t.wait(max_wait_time)
        if t.returncode != 0:
            _logger.error('ERROR - [ {} ] failed with error code {}'.format(
                ' '.join(command_args), t.returncode))
    except ValueError:
        _logger.error('ERROR - Wrong number of arguments passed to subprocess')
        raise ValueError
    except subprocess.TimeoutExpired as s:
        _logger.error('ERROR - Script Timeout during PR creation')
        raise s
    except subprocess.SubprocessError as s:
        _logger.error('ERROR - Some unknown error happened')
        _logger.error('%r' % s)
        raise s


def get_deployed_model_version():
    """Read deployment yaml and return the deployed model verison."""
    # 1. Fetch deployment yaml from saas repo.
    exec_command(['wget', '-v',
                  'https://raw.githubusercontent.com/openshiftio/saas-analytics/'
                  'master/bay-services/f8a-pypi-insights.yaml',
                  '-O', './f8a-pypi-insights.yaml'], 60)

    # 2. Read yaml data and convert to dict
    yaml_dict = ''
    with open('./f8a-pypi-insights.yaml', 'r') as fp:
        yaml_dict = yaml.load(fp.read(), Loader=yaml.FullLoader)

    # 3. Read model version data for given deploment.
    model_version = ''
    environment_name = DEPLOYMENT_PREFIX_ENVIRONMENT_NAME_MAP.get(DEPLOYMENT_PREFIX, '')
    if environment_name:
        environments = yaml_dict.get('services', [{}])[0].get('environments', [])
        for env in environments:
            if env.get('name', '') == environment_name:
                model_version = env.get('parameters', {}).get('MODEL_VERSION', '')
                break
    else:
        _logger.error('ERROR - Invalid deployment prefix: "{}", supported values: {}'.format(
            DEPLOYMENT_PREFIX, list(DEPLOYMENT_PREFIX_ENVIRONMENT_NAME_MAP.keys())))

    _logger.info('Model version: {} for deployment prefix: {}'.format(
        model_version, DEPLOYMENT_PREFIX))
    return model_version


def create_git_pr(s3_client, hyper_params):  # pragma: no cover
    """Create a git PR automatically if recall_at_30 is higher than previous iteration."""
    recall_at_30 = hyper_params['recall_at_30']
    previous_version = get_deployed_model_version()
    k = '{prev_ver}/intermediate-model/hyperparameters.json'.format(prev_ver=previous_version)
    prev_hyperparams = s3_client.read_json_file(k)

    # Convert the json description to string
    description = 'Previous model details:\n' \
                  '    Model version :: {}\n' \
                  '    Hyper parameters :: {}\n\n' \
                  'New model details:\n' \
                  '    Model version :: {}\n' \
                  '    Hyper parameters :: {}\n\n' \
                  'Criteria for promotion was "current_recall_at_30 >= prev_recall_at_30".'.format(
                      previous_version, json.dumps(prev_hyperparams).replace('"', '\\"'),
                      MODEL_VERSION, json.dumps(hyper_params).replace('"', '\\"'))

    prev_recall = prev_hyperparams.get('recall_at_30', 0.55)
    _logger.info('create_git_pr:: Prev => Model {}, Recall {}  Curr => Model {}, Recall {}'.format(
        previous_version, prev_recall, MODEL_VERSION, recall_at_30))
    if recall_at_30 >= prev_recall:
        try:
            # 1. Get the bash script from rudra to raise PR
            exec_command(['wget', '-v',
                          'https://raw.githubusercontent.com/fabric8-analytics/'
                          'fabric8-analytics-rudra/master/rudra/utils/github_helper.sh',
                          '-O', './github_helper.sh'], 60)

            # 2. Provide execute permission to the script file.
            exec_command(['chmod', '+x', './github_helper.sh'], 30)

            # 3. Invoke bash script to create a saas-analytics PR
            exec_command(['./github_helper.sh', 'f8a-pypi-insights.yaml', 'MODEL_VERSION',
                          str(MODEL_VERSION), description], 60)
        except Exception as e:
            _logger.error('ERROR - execute command raise exception')
            raise e
    else:
        _logger.warn('Ignoring latest model {} as its recall {} is less than existing model {} '
                     'recall {}'.format(MODEL_VERSION, recall_at_30, previous_version, prev_recall))


def train_model():
    """Training model."""
    s3_obj = load_s3()
    data = load_data(s3_obj)
    hyper_params = load_hyper_params() or {}
    lower_limit = int(hyper_params.get('lower_limit', 2))
    upper_limit = int(hyper_params.get('upper_limit', 100))
    latent_factors = int(hyper_params.get('latent_factor', 40))
    _logger.info("Deployment type {} Lower limit {}, Upper limit {} and latent factor {} are used."
                 .format(DEPLOYMENT_PREFIX, lower_limit, upper_limit, latent_factors))
    package_id_dict, id_package_dict, manifest_id_dict = \
        preprocess_raw_data(data.get('package_dict', {}), lower_limit, upper_limit)
    user_input_stacks = data.get('package_dict', {}).\
        get('user_input_stack', [])
    user_item_list = make_user_item_df(manifest_id_dict, package_id_dict, user_input_stacks)
    user_item_df = pd.DataFrame(user_item_list)
    training_df, testing_df = train_test_split(user_item_df)
    format_pkg_id_dict, format_mnf_id_dict = format_dict(package_id_dict, manifest_id_dict)
    trained_recommender = run_recommender(training_df, latent_factors)
    precision_at_30, recall_at_30 = precision_recall_at_m(30, testing_df, trained_recommender,
                                                          user_item_df)
    precision_at_50, recall_at_50 = precision_recall_at_m(50, testing_df, trained_recommender,
                                                          user_item_df)
    try:
        hyper_params = build_hyperparams(lower_limit, upper_limit, latent_factors,
                                         precision_at_30, recall_at_30,
                                         precision_at_50, recall_at_50)
        save_obj(s3_obj, trained_recommender, hyper_params,
                 format_pkg_id_dict, id_package_dict, format_mnf_id_dict)
        if GITHUB_TOKEN:
            create_git_pr(s3_client=s3_obj, hyper_params=hyper_params)
        else:
            _logger.info('GITHUB_TOKEN is missing, cannot raise SAAS PR')
    except Exception as error:
        _logger.error(error)
        raise


if __name__ == '__main__':
    train_model()
