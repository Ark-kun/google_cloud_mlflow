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

import docker
import google
import google.auth
from google.cloud.aiplatform import gapic
from mlflow.models import cli
from mlflow.pyfunc import scoring_server
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


def upload_mlflow_model_to_vertex_ai_models(
    model_uri: str,
    display_name: str,
    destination_image_uri: Optional[str] = None,
    model_options: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
    timeout: int = 1800,
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
      timeout: How long to wait for model deployment to complete. Defaults to 30
        minutes.

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
    if not project:
        try:
            _, project = google.auth.default()
            _logger.info("Project not set. Using %s as project", project)
        except google.auth.exceptions.DefaultCredentialsError as e:
            raise ValueError(
                "You must either pass a project ID in or set a default project"
                " (e.g. using gcloud config set project <PROJECT ID>. Default credentials"
                " not found: {}".format(e.message)
            ) from e

    if not destination_image_uri:
        image_name = re.sub("[^-A-Za-z0-9_.]", "_", display_name)
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

    upload_model_response = _upload_model(
        image_uri=pushed_image_uri_with_digest,
        display_name=display_name,
        project=project,
        location=location,
        model_options=model_options,
    ).result(timeout=timeout)
    return upload_model_response.model


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
        "mlflow.pyfunc.backend._build_image",
        new=docker_utils_patch._build_image
    ):
        flavor_backend.build_image(
            model_uri,
            destination_image_uri,
            install_mlflow=mlflow_source_dir is not None,
            mlflow_home=mlflow_source_dir,
        )
    return destination_image_uri
    _logger.info("Uploading image to Google Container Registry")

    client = docker.from_env()
    result = client.images.push(destination_image_uri, stream=True, decode=True)
    for line in result:
        # Docker client doesn't catch auth errors, so we have to do it
        # ourselves. See https://github.com/docker/docker-py/issues/1772
        if "errorDetail" in line:
            raise docker.errors.APIError(line["errorDetail"]["message"])
        if "status" in line:
            _logger.debug(line["status"])
    container_image = client.images.get(destination_image_uri)
    pushed_image_uri_with_digest = container_image.attrs["RepoDigests"][0]
    _logger.info("Uploaded image: %s", pushed_image_uri_with_digest)
    return pushed_image_uri_with_digest


def _upload_model(
    image_uri: str,
    display_name: str,
    model_options: Dict[str, Any],
    project: str,
    location: str,
):
    """Uploads the model with Google Cloud Vertex AI.

    Args:
      image_uri: The URI of the container image for the model.
      display_name: The display name for the Google Cloud Vertex AI Model.
        The name can be up to 128 characters long and can be consist of any UTF-8
        characters.
      model_options: A dict of other attributes of the Google Cloud Vertex AI
        Model object (e.g. labels and schema). See
          [Model](https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1#google.cloud.aiplatform.v1.Model).
      project: The Google Cloud project where to push the container image and
        register the model. If unset, uses the default project from gcloud.
      location: The Google Cloud location where to push the container image and
        register the model. Defaults to "us-central1".

    Returns:
      The full resource name of the Google Cloud Vertex AI Model.
    """
    # Setting environment variables to tell the scoring server to properly wrap
    # the responses.
    # See https://github.com/mlflow/mlflow/pull/4611
    # https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#response_requirements
    env = [
        {
            "name": "PREDICTIONS_WRAPPER_ATTR_NAME",
            "value": "predictions",
        }
    ]
    model_to_upload = {
        "display_name": display_name,
        "container_spec": {
            "image_uri": image_uri,
            "ports": [{"container_port": 8080}],
            "env": env,
            "predict_route": "/invocations",
            "health_route": "/ping",
        },
    }
    if model_options:
        model_to_upload.update(model_options)
    model_to_upload.setdefault("labels", {})[
        "mlflow_model_vertex_ai_deployer"
    ] = "mlflow_model_vertex_ai_deployer"

    client_options = {
        "api_endpoint": f"{location}-aiplatform.googleapis.com",
    }

    model_client = gapic.ModelServiceClient(client_options=client_options)
    model_parent = f"projects/{project}/locations/{location}"
    _logger.info(
        "Uploading model to Google Cloud AI Platform: %s/models/%s",
        model_parent,
        display_name,
    )
    upload_model_response = model_client.upload_model(
        parent=model_parent,
        model=model_to_upload,
    )
    # model: "projects/<project_id>/locations/<location>/models/<model_id>"
    return upload_model_response


def deploy_vertex_ai_model_to_endpoint(
    model_name: str,
    endpoint_name: Optional[str] = None,
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: Optional[int] = None,
    endpoint_display_name: Optional[str] = None,
    deployed_model_display_name: Optional[str] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
    timeout: Optional[float] = None,
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
    # Create an endpoint
    # See https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/create_endpoint_sample.py
    _, default_project = google.auth.default()
    if not project:
        project = default_project
    model_id = model_name.split("/")[-1]

    client_options = {
        "api_endpoint": f"{location}-aiplatform.googleapis.com",
    }
    endpoint_client = gapic.EndpointServiceClient(client_options=client_options)
    if not endpoint_name:
        if not endpoint_display_name:
            endpoint_display_name = model_id
        _logger.info("Creating new Endpoint: %s", endpoint_display_name)
        endpoint_to_create = {
            "display_name": endpoint_display_name,
            "labels": {
                "mlflow_model_vertex_ai_deployer": "mlflow_model_vertex_ai_deployer",
            },
        }
        endpoint = endpoint_client.create_endpoint(
            parent=f"projects/{project}/locations/{location}",
            endpoint=endpoint_to_create,
        ).result(timeout=timeout)
        endpoint_name = endpoint.name
    # projects/<prject_id>/locations/<location>/endpoints/<endpoint_id>
    _logger.info("Endpoint name: %s", endpoint_name)

    # Deploy the model
    # See https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/deploy_model_custom_trained_model_sample.py
    model_to_deploy = {
        "model": model_name,
        "display_name": deployed_model_display_name,
        "dedicated_resources": {
            "min_replica_count": min_replica_count,
            "max_replica_count": max_replica_count,
            "machine_spec": {
                "machine_type": machine_type,
            },
        },
    }
    traffic_split = {"0": 100}
    _logger.info(
        "Deploying model %s to endpoint: %s", model_name, endpoint_display_name
    )
    deploy_model_operation = endpoint_client.deploy_model(
        endpoint=endpoint_name,
        deployed_model=model_to_deploy,
        traffic_split=traffic_split,
    )
    return deploy_model_operation
