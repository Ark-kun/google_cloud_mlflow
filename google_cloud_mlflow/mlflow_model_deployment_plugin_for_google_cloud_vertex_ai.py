"""MLflow deployment plugin to deploy MLflow models to Google Cloud Vertex AI."""

__all__ = [
    "GoogleCloudVertexAiDeploymentClient",
    "target_help",
    "run_local",
]

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from . import _mlflow_model_gcp_deployment_utils as vertex_utils
from google.protobuf import json_format
import mlflow
from mlflow import deployments

# TODO(b/195784726) Remove this workaround once google-cloud-aiplatform conforms
# to the third_party python rules
try:
    from google.cloud import aiplatform  # pylint:disable=g-import-not-at-top
except ImportError:
    from google.cloud.aiplatform import aiplatform  # pylint:disable=g-import-not-at-top

if TYPE_CHECKING:
    # These imports are only used to specify the parameter types and return types
    # for the predict can explain methods.
    import pandas  # pylint:disable=g-import-not-at-top
    import numpy  # pylint:disable=g-import-not-at-top

_logger = logging.getLogger(__name__)

DEFAULT_MACHINE_TYPE="n1-standard-2"


def _resource_to_mlflow_dict(
    resource: aiplatform.base.VertexAiResourceNoun,
) -> Dict[str, Any]:
    """Converts Vertex AI resource instance to a MLflow dict."""
    # TODO(avolkov): Switch to .to_dict() method when my PR is merged:
    # https://github.com/googleapis/python-aiplatform/pull/588
    resource_dict = json_format.MessageToDict(resource._gca_resource._pb)  # pylint: disable=protected-access
    # The MLflow documentation seems to imply that the returned dicts
    # need to have "name" attribute set to MLflow "deployment name", not the
    # internal resource name.
    # We put MLflow deployment name into Endpoint's display_name
    resource_dict["resource_name"] = resource.resource_name
    resource_dict["name"] = resource.display_name
    return resource_dict


class GoogleCloudVertexAiDeploymentClient(deployments.BaseDeploymentClient):
    """The Google Cloud Vertex AI implementation of the BaseDeploymentClient."""

    def create_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: Optional[str] = None,
        config: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Deploys the model.

        Deploys a model to the specified target. By default, this method should
        block until deployment completes (i.e. until it's possible to perform
        inference with the deployment). In the case of conflicts (e.g. if it's
        not possible to create the specified deployment without due to conflict
        with an existing deployment), raises a
        :py:class:`mlflow.exceptions.MlflowException`. See target-specific
        plugin documentation for additional detail on support for asynchronous
        deployment and other configuration.

        Example::

            from mlflow import deployments
            client = deployments.get_deploy_client("google_cloud")
            deployment = client.create_deployment(
                name="deployment name",
                model_uri=...,
                # Config is optional
                config=dict(
                    # Deployed model config
                    machine_type="n1-standard-2",
                    min_replica_count=None,
                    max_replica_count=None,
                    accelerator_type=None,
                    accelerator_count=None,
                    service_account=None,
                    explanation_metadata=None, # JSON string
                    explanation_parameters=None, # JSON string

                    # Model container image building config
                    destination_image_uri=None,
                    timeout=None,

                    # Model deployment config
                    sync="true",

                    # Endpoint config
                    description=None,

                    # Vertex AI config
                    project=None,
                    location=None,
                    experiment=None,
                    experiment_description=None,
                    staging_bucket=None,
                )
            )

        Args:
            name: Unique name to use for deployment. If another deployment
                exists with the same name, raises a
                :py:class:`mlflow.exceptions.MlflowException`
            model_uri: URI of model to deploy
            flavor: (optional) The MLflow model flavor to deploy.
                If unspecified, the default flavor will be chosen.
            config: (optional) Dict containing updated target-specific configuration for the
                deployment

        Returns:
            A dict corresponding to the created deployment, which must contain the 'name' key.
        """
        config = config or {}
        existing_endpoints = aiplatform.Endpoint.list(filter=f'display_name="{name}"')
        if existing_endpoints:
            raise mlflow.exceptions.MlflowException(
                f"Found existing deployment with name {name}: " +
                ",".join(list(endpoint.resource_name for endpoint in existing_endpoints)))

        aiplatform.init(
            project=config.get("project"),
            location=config.get("location"),
            experiment=config.get("experiment"),
            experiment_description=config.get("experiment_description"),
            staging_bucket=config.get("staging_bucket"),
            encryption_spec_key_name=config.get("encryption_spec_key_name"),
        )
        model_name = config.get("model_resource_name")
        if not model_name:
            model_name = vertex_utils.upload_mlflow_model_to_vertex_ai_models(
                model_uri=model_uri,
                display_name=name,
                destination_image_uri=config.get("destination_image_uri"),
                timeout=int(config.get("timeout", 1800)),
            )
        endpoint = aiplatform.Endpoint.create(
            display_name=name,
            description=config.get("description"),
            encryption_spec_key_name=config.get("encryption_spec_key_name"),
            labels={
                "google_cloud_mlflow_plugin_version": "0.0.1",
            },
        )
        endpoint.deploy(
            model=aiplatform.Model(model_name=model_name),
            deployed_model_display_name=name,
            traffic_percentage=100,
            # Need to always specify the machine type to prevent this error:
            # InvalidArgument: 400 'automatic_resources' is not supported for Model
            # The choice between "dedicated_resources" and "automatic_resources"
            # (only supported with AutoML models) is based on the presence of
            # machine_type.
            machine_type=config.get("machine_type", DEFAULT_MACHINE_TYPE),
            min_replica_count=int(config.get("min_replica_count", 1)),
            max_replica_count=int(config.get("max_replica_count", 1)),
            accelerator_type=config.get("accelerator_type"),
            accelerator_count=int(config.get("accelerator_count", 0)) or None,
            service_account=config.get("service_account"),
            explanation_metadata=(json.loads(config.get("explanation_metadata")) if "explanation_metadata" in config else None),
            explanation_parameters=(json.loads(config.get("explanation_parameters")) if "explanation_parameters" in config else None),
            sync=json.loads(config.get("sync", "true")),
        )

        deployment_dict = _resource_to_mlflow_dict(endpoint)
        deployment_dict["flavor"] = flavor
        return deployment_dict

    def get_deployment(self, name: str) -> Dict[str, Any]:
        """Gets deployment by name.

        Args:
            name: ID of deployment to fetch

        Returns:
            A dictionary describing the specified deployment, throwing a
            py:class:`mlflow.exception.MlflowException` if no deployment exists with the provided ID.
            The dict is guaranteed to contain an 'name' key containing the deployment name.
            The other fields of the returned dictionary and their types may vary across
            deployment targets.
        """
        return _resource_to_mlflow_dict(
            self._get_deployment(deployment_name=name)
        )
    
    def _get_deployment(self, deployment_name: str) -> aiplatform.Endpoint:
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{deployment_name}"')
        if len(endpoints) > 1:
            raise mlflow.exceptions.MlflowException(
                f"Found multiple deployments with name {deployment_name}: " +
                ",".join(list(endpoint.resource_name for endpoint in endpoints)))
        if endpoints:
            return endpoints[0]
        raise mlflow.exceptions.MlflowException(
            f"Could not find deployment with name {deployment_name}."
        )

    def list_deployments(self) -> List[Dict[str, Any]]:
        """Lists all deployments.

        Returns:
            A list of dicts corresponding to deployments. Each dict is guaranteed to
            contain a 'name' key containing the deployment name. The other fields of
            the returned dictionary and their types may vary across deployment targets.
        """
        endpoints = aiplatform.Endpoint.list(filter='labels.google_cloud_mlflow_plugin_version:*')
        endpoint_dicts = list(map(_resource_to_mlflow_dict, endpoints))
        return endpoint_dicts

    def delete_deployment(self, name: str) -> None:
        """Deletes the deployment.

        Deletes the deployment with name ``name`` from the specified target.
        Deletion is idempotent (i.e. deletion does not fail if retried on a
        non-existent deployment).

        Args:
            name: The name of deployment to delete
        """
        deployment_name = name
        # Locate deployment endpoint from MLflow deployment list
        # Using Endpoint.delete with force=True will undeploy all models on Endpoint
        # before deleting the Endpoint
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{deployment_name}"')
        if len(endpoints) > 1:
            raise mlflow.exceptions.MlflowException(
                f"Found multiple deployments with name {deployment_name}: " +
                ",".join(list(endpoint.resource_name for endpoint in endpoints)))
        if endpoints:
            endpoint = endpoints[0]
            endpoint.delete(force=True)

    def update_deployment(
        self,
        name: str,
        model_uri: Optional[str] = None,
        flavor: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Updates the deployment with the specified name.

        You can update the URI of the model, the flavor of the deployed model
        (in which case the model URI must also be specified), and/or any
        target-specific attributes of the deployment (via `config`). By default,
        this method should block until deployment completes (i.e. until it's
        possible to perform inference with the updated deployment).
        See target-specific plugin documentation for additional detail on support
        for asynchronous deployment and other configuration.

        Args:
            name: Unique name of deployment to update
            model_uri: URI of a new model to deploy.
            flavor: (optional) new model flavor to use for deployment. If provided,
                ``model_uri`` must also be specified. If ``flavor`` is unspecified but
                ``model_uri`` is specified, a default flavor will be chosen and the
                deployment will be updated using that flavor.
            config: (optional) dict containing updated target-specific configuration for the
                deployment

        Returns:
            A dict corresponding to the created deployment, which must contain the 'name' key.
        """
        self.delete_deployment(name=name)
        return self.create_deployment(
            name=name,
            model_uri=model_uri,
            flavor=flavor,
            config=config,
        )

    def predict(
        self, deployment_name: str, df: "pandas.DataFrame"
    ) -> Union["pandas.DataFrame", "pandas.Series", "numpy.ndarray", Dict[str, Any]]:
        """Computes model predictions.

        Compute predictions on the pandas DataFrame ``df`` using the specified
        deployment. Note that the input/output types of this method matches that
        of `mlflow pyfunc predict` (we accept a pandas DataFrame as input and
        return either a pandas DataFrame, pandas Series, or numpy array as output).

        Args:
            deployment_name: Name of deployment to predict against
            df: Pandas DataFrame to use for inference

        Returns:
            A pandas DataFrame, pandas Series, or numpy array

        Example::

            from mlflow import deployments
            import pandas
            df = pandas.DataFrame(
                [
                    {"a": 1,"b": 2,"c": 3},
                    {"a": 4,"b": 5,"c": 6}
                ]
            )
            client = deployments.get_deploy_client("google_cloud")
            client.create_deployment("deployment name", model_uri=...)
            predictions = client.predict("deployment name", df)
        """
        endpoint = self._get_deployment(deployment_name=deployment_name)
        predictions = endpoint.predict(
            instances=df.to_dict("records")
        )
        return predictions

    def explain(self, deployment_name: str, df: "pandas.DataFrame"):
        """Generates explanations of model predictions.

        Generate explanations of model predictions on the specified input pandas Dataframe
        ``df`` for the deployed model. Explanation output formats vary by deployment target,
        and can include details like feature importance for understanding/debugging predictions.

        Args:
            deployment_name: Name of deployment to predict against
            df: Pandas DataFrame to use for inference

        Returns:
            A JSON-able object (pandas dataframe, numpy array, dictionary), or
            an exception if the implementation is not available in deployment target's class

        Example::

            from mlflow import deployments
            import pandas
            df = pandas.DataFrame(
                [
                    {"a": 1,"b": 2,"c": 3},
                    {"a": 4,"b": 5,"c": 6}
                ]
            )
            client = deployments.get_deploy_client("google_cloud")
            client.create_deployment("deployment name", model_uri=...)
            predictions = client.explain("deployment name", df)
        """
        endpoint = self._get_deployment(deployment_name=deployment_name)
        predictions = endpoint.explain(
            instances=df.to_dict("records")
        )
        return predictions


def run_local(
    name: str,
    model_uri: str,
    flavor: Optional[str] = None,
    config: Optional[Dict[str, str]] = None,
):
    """Deploys the specified model locally, for testing.

    Args:
        name: Unique name to use for deployment. If another deployment exists with
            the same name, create_deployment will raise a
            :py:class:`mlflow.exceptions.MlflowException`
        model_uri: URI of model to deploy
        flavor: Model flavor to deploy. If unspecified, default flavor is chosen.
        config: Dict containing updated target-specific config for the deployment
    """
    raise NotImplementedError()


def target_help():
    """Returns help string.

    Returns a string containing detailed documentation on the current deployment
    target, to be displayed when users invoke the
    ``mlflow deployments help -t <target-name>`` CLI command.
    """
    return """
        MLflow deployment plugin to deploy MLflow models to Google Cloud Vertex AI.

        Example::

            from mlflow import deployments
            client = deployments.get_deploy_client("google_cloud")
            deployment = client.create_deployment(
                name="deployment name",
                model_uri=...,
                # Config is optional
                config=dict(
                    # Deployed model config
                    machine_type="n1-standard-2",
                    min_replica_count=None,
                    max_replica_count=None,
                    accelerator_type=None,
                    accelerator_count=None,
                    service_account=None,
                    explanation_metadata=None, # JSON string
                    explanation_parameters=None, # JSON string

                    # Model container image building config
                    destination_image_uri=None,
                    timeout=None,

                    # Model deployment config
                    sync="true",

                    # Endpoint config
                    description=None,

                    # Vertex AI config
                    project=None,
                    location=None,
                    experiment=None,
                    experiment_description=None,
                    staging_bucket=None,
                )
            )
    """
