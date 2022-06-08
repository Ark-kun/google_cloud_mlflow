# Copyright 2021 The google-cloud-mlflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This module provides an API for deploying MLflow models to Google Cloud Vertex AI.

The `upload_mlflow_model_to_vertex_ai_models` function builds a Docker container
for a given MLflow model and pushes the image to Google Container Registry.
Then it registers the model as a Google Cloud Vertex AI Model.
Once the model is registered, the user can deploy it for serving on Google Cloud
Vertex AI Endpoint using the `deploy_gcp_model_to_new_endpoint` function.
See
[docs](https://cloud.google.com/ai-platform-unified/docs/predictions/deploy-model-api)
for more information.

Examples::

    # Use MLflow to register the model on Cloud AI Platform
    model_uri = "models:/mymodel/mymodelversion" # Replace with your model URI
    display_name = "my_mlflow_model" # Replace with the desired model name

    model_name = upload_mlflow_model_to_vertex_ai_models(
        model_uri=model_uri,
        display_name=display_name,
    )

    deploy_model_operation = deploy_vertex_ai_model_to_endpoint(
        model_name=model_name,
    )
    deployed_model = deploy_model_operation.result().deployed_model
"""

import logging
import os
import re
import tempfile
from typing import Any, Dict, Optional
import urllib
import zipfile

import google
from google.cloud import aiplatform
from mlflow.models import cli, Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from unittest import mock

from . import _mlflow_models_docker_utils_patch as docker_utils_patch


_logger = logging.getLogger(__name__)


def get_fixed_mlflow_source_dir():
    """Downloads the fixed MLflow source code."""
    fixed_mlflow_archive_url = "https://github.com/Ark-kun/mlflow/archive/refs/heads/MLFlow-fixes.zip"
    fixed_mlflow_archive_path, _ = urllib.request.urlretrieve(url=fixed_mlflow_archive_url)
    fixed_mlflow_parent_dir = tempfile.mkdtemp(prefix="mlflow.fixed")
    with zipfile.ZipFile(fixed_mlflow_archive_path, 'r') as zip_ref:
        zip_ref.extractall(fixed_mlflow_parent_dir)
    # The archive contains a subdirectory: "Ark-kun-mlflow-0ec4c64"
    # So we need to go one level deeper
    subdir = os.listdir(fixed_mlflow_parent_dir)[0]
    fixed_mlflow_dir = os.path.join(fixed_mlflow_parent_dir, subdir)
    return fixed_mlflow_dir


def get_pickle_protocol(file_path: str) -> int:
    import pickletools

    max_proto = -1
    with open(file_path, "rb") as file:
        try:
            for opcode, arg, _ in pickletools.genops(file):
                if opcode.name == "PROTO":
                    return arg
                # Looking at the opcode.proto is not reliable since unsupported opcodes cannot be parsed by old python versions.
                max_proto = max(max_proto, opcode.proto)
        except:
            pass
    return max_proto
    

def upload_mlflow_model_to_vertex_ai_models(
    model_uri: str,
    display_name: str,
    destination_image_uri: Optional[str] = None,
    model_options: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    encryption_spec_key_name: Optional[str] = None,
    staging_bucket: Optional[str] = None,
) -> str:
    """Builds a container for an MLflow model and registers the model with Google Cloud Vertex AI.

    The resulting container image will contain the MLflow webserver that processes
    prediction requests. The container image can be deployed as a web service to
    Vertex AI Endpoints.

    Args:
      model_uri: The URI of the MLflow model.
        Format examples:
          * `/Users/me/path/to/local/model`
          * `relative/path/to/local/model`
          * `gs://my_bucket/path/to/model`
          * `runs:/<mlflow_run_id>/run-relative/path/to/model`
          * `models:/<model_name>/<model_version>`
          * `models:/<model_name>/<stage>`
          For more information about supported URI schemes, see [Referencing
          Artifacts](https://www.mlflow.org/docs/latest/concepts.html#artifact-locations).
      display_name: The display name for the Google Cloud Vertex AI Model.
        The name can be up to 128 characters long and can be consist of any UTF-8
        characters.
      destination_image_uri: The full name of the container image that will be
        built with the provided model inside it.
        The format should be `gcr.io/<REPO>/<IMAGE>:<TAG>`.
        Defaults to `gcr.io/<DEFAULT_PROJECT>/mlflow/<display_name>:<LATEST>`
      model_options: A dict of other attributes of the Google Cloud Vertex AI
        Model object, like labels and schema. See
          [Model](https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1#google.cloud.aiplatform.v1.Model).
      project: The Google Cloud project where to push the container image
        and register the model. Defaults to the location used by the gcloud CLI.
      location: The Google Cloud location where to push the container image
        and register the model. Defaults to "us-central1".
      encryption_spec_key_name:
        Optional. The Cloud KMS resource identifier of the customer
        managed encryption key used to protect the model. Has the
        form:
        ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
        The key needs to be in the same region as where the compute
        resource is created.

        If set, this Model and all sub-resources of this Model will be secured
        by this key.

        Overrides encryption_spec_key_name set in aiplatform.init.
      staging_bucket: Optional. Bucket to stage local model artifacts.
        Overrides staging_bucket set in aiplatform.init.

    Returns:
      The full resource name of the Google Cloud Vertex AI Model.

    Examples::

        # Use MLflow to register the model on Google Cloud Vertex AI
        model_uri = "models:/mymodel/mymodelversion" # Replace with your model URI
        display_name = "my_mlflow_model" # Replace with the desired model name

        model_name = upload_mlflow_model_to_vertex_ai_models(
            model_uri=model_uri,
            display_name=display_name,
        )

        deployed_model_id = deploy_vertex_ai_model_to_endpoint(
            model_name=model_name,
        )
    """
    project = project or aiplatform.initializer.global_config.project

    temp_dir = tempfile.mkdtemp()
    model_dir = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=temp_dir,
    )
    model_metadata = Model.load(os.path.join(model_dir, MLMODEL_FILE_NAME))

    for flavor_name, flavor in model_metadata.flavors.items():
        if flavor_name == "python_function":
            continue
        if flavor_name == "xgboost":
            model_file_name = flavor["data"]
            full_xgboost_version = flavor["xgb_version"]
            model_file_path = os.path.join(model_dir, model_file_name)
            # TODO: Remove after https://b.corp.google.com/issues/216705259 is fixed
            pickle_protocol = get_pickle_protocol(model_file_path)
            # Vertex Prediction uses Python 3.7 which does not support pickle protocol 5
            if pickle_protocol == 5:
                _logger.warning("Detected model with pickle protocol version 5 > 4. Prebuilt containers do not support such models.")
                continue
            # TODO: Handle case when the version is not supported by Vertex AI
            vertex_xgboost_version = ".".join(full_xgboost_version.split(".")[0:2])
            vertex_model = aiplatform.Model.upload_xgboost_model_file(
                model_file_path=model_file_path,
                xgboost_version=vertex_xgboost_version,
                display_name=display_name,
                project=project,
                location=location,
                encryption_spec_key_name=encryption_spec_key_name,
                staging_bucket=staging_bucket,
            )
            return vertex_model.resource_name
        if flavor_name == "sklearn":
            model_file_name = flavor["pickled_model"]
            model_file_path = os.path.join(model_dir, model_file_name)
            # TODO: Remove after https://b.corp.google.com/issues/216705259 is fixed
            pickle_protocol = get_pickle_protocol(model_file_path)
            # Vertex Prediction uses Python 3.7 which does not support pickle protocol 5
            if pickle_protocol == 5:
                _logger.warning("Detected model with pickle protocol version 5 > 4. Prebuilt containers do not support such models.")
                continue
            vertex_model = aiplatform.Model.upload_scikit_learn_model_file(
                model_file_path=model_file_path,
                # TODO: Deduce version from requirements.txt
                # sklearn_version=
                display_name=display_name,
                project=project,
                location=location,
                encryption_spec_key_name=encryption_spec_key_name,
                staging_bucket=staging_bucket,
            )
            return vertex_model.resource_name
        if flavor_name == "tensorflow":
            model_dir_name = flavor["saved_model_dir"]
            model_dir_path = os.path.join(model_dir, model_dir_name)
            vertex_model = aiplatform.Model.upload_tensorflow_saved_model(
                saved_model_dir=model_dir_path,
                # TODO: Deduce version from requirements.txt
                # tensorflow_version=
                display_name=display_name,
                project=project,
                location=location,
                encryption_spec_key_name=encryption_spec_key_name,
                staging_bucket=staging_bucket,
            )
            return vertex_model.resource_name

    _logger.info(
        "Model flavor is not directly supported by Vertex AI. Importing model as a custom-built container"
    )

    if not destination_image_uri:
        image_name = re.sub("[^-A-Za-z0-9_.]", "_", display_name).lower()
        destination_image_uri = f"gcr.io/{project}/mlflow/{image_name}"
        _logger.info(
            "Destination image URI not set. Building and uploading image to %s",
            destination_image_uri,
        )

    pushed_image_uri_with_digest = _build_serving_image(
        model_uri=model_uri,
        destination_image_uri=destination_image_uri,
        mlflow_source_dir=None,
    )

    uploaded_model = aiplatform.Model.upload(
        # artifact_uri=
        display_name=display_name,
        # description=
        serving_container_image_uri=pushed_image_uri_with_digest,
        # serving_container_command=
        # serving_container_args=
        serving_container_predict_route="/invocations",
        serving_container_health_route="/ping",
        # Setting environment variables to tell the scoring server to properly wrap
        # the responses.
        # See https://github.com/mlflow/mlflow/pull/4611
        # https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#response_requirements
        serving_container_environment_variables={
            "PREDICTIONS_WRAPPER_ATTR_NAME": "predictions",
        },
        serving_container_ports=[8080],
        project=project,
        location=location,
        labels={
            "mlflow_model_vertex_ai_deployer": "mlflow_model_vertex_ai_deployer",
        },
        encryption_spec_key_name=encryption_spec_key_name,
        staging_bucket=staging_bucket,
    )
    return uploaded_model.resource_name


def _build_serving_image(
    model_uri: str,
    destination_image_uri: str,
    mlflow_source_dir: Optional[str] = None,
) -> str:
    """Builds and pushes an MLflow serving image for the MLflow model.

    Args:
      model_uri: The URI of the MLflow model.
      destination_image_uri: The full name of the container image that will
        be built with the provided model inside it.
        The format should be `gcr.io/<REPO>/<IMAGE>:<TAG>`.
      mlflow_source_dir: If set, installs MLflow from this directory instead of
        PyPI.
    Returns:
      Fully-qualified URI of the pushed container image including the hash digest.
    """
    _logger.info("Building image. This can take up to 20 minutes")
    flavor_backend = cli._get_flavor_backend(
        model_uri
    )  # pylint:disable=protected-access

    with mock.patch(
        "mlflow.models.docker_utils._build_image_from_context",
        new=docker_utils_patch._build_image_from_context
    ):
        flavor_backend.build_image(
            model_uri,
            destination_image_uri,
            install_mlflow=mlflow_source_dir is not None,
            mlflow_home=mlflow_source_dir,
        )
    return destination_image_uri


def deploy_vertex_ai_model_to_endpoint(
    model_name: str,
    endpoint_name: Optional[str] = None,
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    endpoint_display_name: Optional[str] = None,
    deployed_model_display_name: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> google.api_core.operation.Operation:
    # pylint: disable=line-too-long
    """Deploys Google Cloud Vertex AI Model to a Google Cloud Vertex AI Endpoint.

    Args:
      model_name: Full resource name of a Google Cloud Vertex AI Model
      endpoint_name: Full name of Google Cloud Vertex Endpoint. A new
        enpoint is created if the name is not passed.
      machine_type: The type of the machine. See the [list of machine types
        supported for
        prediction](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types).
        Defaults to "n1-standard-2"
      min_replica_count: The minimum number of replicas the DeployedModel
        will be always deployed on. If traffic against it increases, it may
        dynamically be deployed onto more replicas up to max_replica_count, and as
        traffic decreases, some of these extra replicas may be freed. If the
        requested value is too large, the deployment will error. Defaults to 1.
      max_replica_count: The maximum number of replicas this DeployedModel
        may be deployed on when the traffic against it increases. If the requested
        value is too large, the deployment will error, but if deployment succeeds
        then the ability to scale the model to that many replicas is guaranteed
        (barring service outages). If traffic against the DeployedModel increases
        beyond what its replicas at maximum may handle, a portion of the traffic
        will be dropped. If this value is not provided, a no upper bound for
        scaling under heavy traffic will be assume, though Vertex AI may be unable
        to scale beyond certain replica number. Defaults to `min_replica_count`
      endpoint_display_name: The display name of the Endpoint. The name can
        be up to 128 characters long and can be consist of any UTF-8 characters.
        Defaults to the lowercased model ID.
      deployed_model_display_name: The display name of the DeployedModel. If
        not provided upon creation, the Model's display_name is used.
      project: The Google Cloud project ID. Defaults to the default project.
      location: The Google Cloud region. Defaults to "us-central1"
      timeout: Model deployment timeout

    Returns:
        google.api_core.operation.Operation:
            An object representing a long-running operation.

            The result type for the operation will be
            :class:`google.cloud.aiplatform_v1.types.DeployModelResponse`
            Response message for
            [EndpointService.DeployModel][google.cloud.aiplatform.v1.EndpointService.DeployModel]
            See
            https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1#google.cloud.aiplatform.v1.DeployedModel

    Examples::

        # Use MLflow to register the model on Cloud AI Platform
        model_uri = "models:/mymodel/mymodelversion" # Replace with your model URI
        display_name = "my_mlflow_model" # Replace with the desired model name

        model_name = upload_mlflow_model_to_vertex_ai_models(
            model_uri=model_uri,
            display_name=display_name,
        )

        deployed_model_id = deploy_vertex_ai_model_to_endpoint(
            model_name=model_name,
        )
    """
    model = aiplatform.Model(model_name)
    if endpoint_name:
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
    else:
        # Model.deploy can create the Endpoint automatically, but I want to add label to the endpoint.
        if not endpoint_display_name:
            endpoint_display_name = model.display_name[:127]
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            labels={
                "mlflow_model_vertex_ai_deployer": "mlflow_model_vertex_ai_deployer",
            },
            project=project,
            location=location,
        )

    return model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=100,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )
