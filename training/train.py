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
import ruamel.yaml
from github import Github

from src.config.path_constants import (PACKAGE_TO_ID_MAP, ID_TO_PACKAGE_MAP,
                                       MANIFEST_TO_ID_MAP, MANIFEST_PATH, HPF_MODEL_PATH, ECOSYSTEM,
                                       HYPERPARAMETERS_PATH, MODEL_VERSION)
from src.config.cloud_constants import (AWS_S3_BUCKET_NAME, AWS_S3_SECRET_KEY_ID,
                                        AWS_S3_ACCESS_KEY_ID, GITHUB_TOKEN, DEPLOYMENT_PREFIX)

logging.basicConfig()
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

bq_validator = BQValidation()

UPSTREAM_REPO_NAME = 'openshiftio'
FORK_REPO_NAME = 'developer-analytics-bot'
PROJECT_NAME = 'saas-analytics'
YAML_FILE_PATH = 'bay-services/f8a-pypi-insights.yaml'


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
                      precision_30, recall_30, precision_50, recall_50, deployment_type):
    """Build hyper parameter object."""
    return {
        "deployment": deployment_type,
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


def get_deployed_model_version(yaml_dict, deployment_type):
    """Read deployment yaml and return the deployed model verison."""
    model_version = None
    environments = yaml_dict.get('services', [{}])[0].get('environments', [])
    for env in environments:
        if env.get('name', '') == deployment_type:
            model_version = env.get('parameters', {}).get('MODEL_VERSION', '')
            break

    if model_version is None:
        raise Exception(f'Model version could not be found for deployment {deployment_type}')

    _logger.info('Model version: %s for deployment: %s', model_version, deployment_type)
    return model_version


def update_yaml_data(yaml_dict, deployment_type, model_version, hyper_params):
    """Update the yaml file for given deployment with model data and description as comments."""
    environments = yaml_dict.get('services', [{}])[0].get('environments', [])
    hyper_params = { k: str(v) for k, v in hyper_params.items()}
    for index, env in enumerate(environments):
        if env.get('name', '') == deployment_type:
            yaml_dict['services'][0]['environments'][index]['comments'] = hyper_params
            yaml_dict['services'][0]['environments'][index]['parameters']['MODEL_VERSION'] = \
                model_version
            break

    return ruamel.yaml.dump(yaml_dict, Dumper=ruamel.yaml.RoundTripDumper)


def build_hyper_params_message(hyper_params):
    """Build hyper params data string used for PR description and in yaml comments."""
    return '- Hyper parameters :: {}'.format(json.dumps(hyper_params, indent=4, sort_keys=True))


def format_body(body):
    """Format PR body string to replace decorators."""
    return body.replace('"', '').replace('{', '').replace('}', '').replace(',', '')


def read_deployed_data(upstream_repo, s3_client, deployment_type):
    """Read deployed data like yaml file, hyper params, model version."""
    upstream_latest_commit_hash = upstream_repo.get_commits()[0].sha
    _logger.info('Upstream latest commit hash: %s', upstream_latest_commit_hash)

    contents = upstream_repo.get_contents(YAML_FILE_PATH, ref=upstream_latest_commit_hash)
    yaml_dict = ruamel.yaml.load(contents.decoded_content.decode('utf8'),
                                 ruamel.yaml.RoundTripLoader)

    deployed_version = get_deployed_model_version(yaml_dict, deployment_type)
    deployed_file_path = f'{deployed_version}/intermediate-model/hyperparameters.json'
    deployed_hyperparams = s3_client.read_json_file(deployed_file_path)

    deployed_data = {
        'version': deployed_version,
        'hyperparams': deployed_hyperparams
    }
    yaml_data = {
        'content_sha': contents.sha,
        'dict': yaml_dict
    }

    return deployed_data, yaml_data, upstream_latest_commit_hash


def create_branch_and_update_yaml(deployment_type, deployed_data, yaml_data,
                                  hyper_params, latest_commit_hash):
    """Create branch and update yaml content on fork repo."""
    # Update yaml model version for the given deployment
    new_yaml_data = update_yaml_data(yaml_data['dict'], deployment_type,
                                     MODEL_VERSION, hyper_params)
    _logger.info('Modified yaml data, new length: %d', len(new_yaml_data))

    # Connect to fabric8 analytic repo & get latest commit hash
    f8a_repo = Github(GITHUB_TOKEN).get_repo(f'{FORK_REPO_NAME}/{PROJECT_NAME}')
    _logger.info('f8a fork repo: %s', f8a_repo)

    # Create a new branch on f8a repo
    branch_name = f'bump_f8a-pypi-insights_for_{deployment_type}_to_{MODEL_VERSION}'
    branch = f8a_repo.create_git_ref(f'refs/heads/{branch_name}', latest_commit_hash)
    _logger.info('Created new branch [%s] at [%s]', branch, latest_commit_hash)

    # Update the yaml content in branch on f8a repo
    commit_message = f'Bump up f8a-pypi-insights for {deployment_type} from ' \
                     f'{deployed_data["version"]} to {MODEL_VERSION}'
    update = f8a_repo.update_file(YAML_FILE_PATH, commit_message, new_yaml_data,
                                  yaml_data['content_sha'], branch=f'refs/heads/{branch_name}')
    _logger.info('New yaml content hash %s', update['commit'].sha)

    return branch_name, commit_message


def create_git_pr(s3_client, hyper_params, deployment_type):  # pragma: no cover
    """Create a git PR automatically if recall_at_30 is higher than previous iteration."""
    upstream_repo = Github(GITHUB_TOKEN).get_repo(f'{UPSTREAM_REPO_NAME}/{PROJECT_NAME}')
    deployed_data, yaml_data, latest_commit_hash = read_deployed_data(upstream_repo, s3_client,
                                                                      deployment_type)

    recall_at_30 = hyper_params['recall_at_30']
    deployed_recall_at_30 = deployed_data['hyperparams'].get('recall_at_30', 0.55)
    _logger.info('create_git_pr:: Deployed => Model %s, Recall %f Current => Model %s, Recall %f',
                 deployed_data['version'], deployed_recall_at_30, MODEL_VERSION, recall_at_30)
    if recall_at_30 >= deployed_recall_at_30:
        promotion_creteria = 'current_recall_at_30 >= deployed_recall_at_30'

        params = hyper_params.copy()
        params.update({'promotion_criteria': str(promotion_creteria)})
        branch_name, commit_message = create_branch_and_update_yaml(deployment_type, deployed_data,
                                                                    yaml_data, params,
                                                                    latest_commit_hash)

        hyper_params_formated = build_hyper_params_message(hyper_params)
        prev_hyper_params_formated = build_hyper_params_message(deployed_data['hyperparams'])
        body = f'''Current deployed model details:
- Model version :: `{deployed_data['version']}`
{prev_hyper_params_formated}

New model details:
- Model version :: `{MODEL_VERSION}`
{hyper_params_formated}

Criteria for promotion is `{promotion_creteria}`
'''
        pr = upstream_repo.create_pull(title=commit_message, body=format_body(body),
                                       head=f'{FORK_REPO_NAME}:{branch_name}',
                                       base='refs/heads/main')
        _logger.info('Raised SAAS %s for review', pr)
    else:
        _logger.warn('Ignoring latest model %s as its recall %f is less than '
                     'existing model %s recall %f', MODEL_VERSION, recall_at_30,
                     deployed_data['version'], deployed_recall_at_30)


def train_model():
    """Training model."""
    deployment_prefix_to_type_map = {
        'STAGE': 'staging',
        'PROD': 'production'
    }

    deployment_type = deployment_prefix_to_type_map.get(DEPLOYMENT_PREFIX, None)
    assert deployment_type is not None, f'Invalid DEPLOYMENT_PREFIX: {DEPLOYMENT_PREFIX}'

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
                                         precision_at_50, recall_at_50, deployment_type)
        save_obj(s3_obj, trained_recommender, hyper_params,
                 format_pkg_id_dict, id_package_dict, format_mnf_id_dict)
        if GITHUB_TOKEN:
            create_git_pr(s3_obj, hyper_params, deployment_type)
        else:
            _logger.info('GITHUB_TOKEN is missing, cannot raise SAAS PR')
    except Exception as error:
        _logger.error(error)
        raise


if __name__ == '__main__':
    train_model()
