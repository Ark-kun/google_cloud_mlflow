# Copyright 2023 The google-cloud-mlflow Authors.
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
"""MLflow Model Registry plugin that allows logging MLFlow models to Google Cloud Vertex AI."""

from typing import List, Optional

import mlflow
from mlflow.entities import model_registry
from mlflow.entities.model_registry import model_version_stages
from mlflow.protos import databricks_pb2
from mlflow.store.entities import paged_list
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils import validation

from google.cloud import aiplatform

# Cannot use dots in the key name since it interferes with the filtering syntax.
# See https://cloud.google.com/vertex-ai/docs/ml-metadata/analyzing#filter_on_metadata_using_the_traversal_operator
_MLFLOW_MODEL_REGISTRY_PLUGIN_VERSION_KEY = (
    "google-cloud-mlflow-model-registry-plugin-version"
)
_MLFLOW_MODEL_REGISTRY_PLUGIN_VERSION_VALUE = "0.0.1"

_MLFLOW_MODEL_KEY = "google-cloud-mlflow-model"
_MLFLOW_MODEL_VALUE = "true"
_MLFLOW_MODEL_NAME_KEY = "google-cloud-mlflow-model-name"
_MLFLOW_MODEL_DESCRIPTION_KEY = "google-cloud-mlflow-model-description"
_MLFLOW_MODEL_TAGS_KEY = "google-cloud-mlflow-model-tags"
_MLFLOW_MODEL_NEXT_VERSION_KEY = "google-cloud-mlflow-model-next-version"

_MLFLOW_MODEL_VERSION_KEY = "google-cloud-mlflow-model-version"
_MLFLOW_MODEL_VERSION_VALUE = "true"
_MLFLOW_MODEL_VERSION_NAME_KEY = "google-cloud-mlflow-model-version-name"
_MLFLOW_MODEL_VERSION_VERSION_KEY = "google-cloud-mlflow-model-version-version"
_MLFLOW_MODEL_VERSION_PARENT_MODEL_ARTIFACT_NAME_KEY = (
    "google-cloud-mlflow-model-version-parent-model-artifact-name"
)
_MLFLOW_MODEL_VERSION_STAGE_KEY = "google-cloud-mlflow-model-version-stage"
# _MLFLOW_MODEL_VERSION_OBJECT_KEY = "google-cloud-mlflow-model-version-object"
_MLFLOW_MODEL_VERSION_DESCRIPTION_KEY = "google-cloud-mlflow-model-version-description"
_MLFLOW_MODEL_VERSION_TAGS_KEY = "google-cloud-mlflow-model-version-tags"
_MLFLOW_MODEL_VERSION_SOURCE_KEY = "google-cloud-mlflow-model-version-source"
_MLFLOW_MODEL_VERSION_RUN_ID_KEY = "google-cloud-mlflow-model-version-run-id"
_MLFLOW_MODEL_VERSION_RUN_LINK_KEY = "google-cloud-mlflow-model-version-run-link"


class GoogleCloudVertexAIModelRegistry(
    mlflow.store.model_registry.abstract_store.AbstractStore
):
    """Class for storing Model Registry metadata."""

    def __init__(self, store_uri: Optional[str] = None):
        # store_uri = "google-cloud-vertex-ai://projects/.../locations/.../metadataStores/default"
        if store_uri:
            if store_uri.starts_with("google-cloud-vertex-ai://"):
                _, _, _, project, _, location, _, metadata_store_id = store_uri.split("/")
                self._project = project
                self._location = location
                self._metadata_store_id = metadata_store_id
            else:
                raise mlflow.exceptions.MlflowException(
                    f"Store URI should have 'google-cloud-vertex-ai://projects/.../locations/.../metadataStores/...' format, but got: {store_uri}"
                )
        self._project = aiplatform.initializer.global_config.project
        self._location = aiplatform.initializer.global_config.location
        self._metadata_store_id = "default"

    # CRUD API for RegisteredModel objects
    def create_registered_model(
        self,
        name: str,
        tags: List[model_registry.RegisteredModelTag] = None,
        description: str = None,
    ) -> model_registry.RegisteredModel:
        """Creates a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.
        """
        validation._validate_model_name(name)
        for tag in tags or []:
            validation._validate_registered_model_tag(tag.key, tag.value)

        # Verifying that the model does not exist yet
        try:
            self.get_registered_model(name=name)
            raise mlflow.MlflowException(
                f"The model with name={name} already exists",
                databricks_pb2.RESOURCE_ALREADY_EXISTS,
            )
        except mlflow.MlflowException as ex:
            # MlflowException converts error codes o strings O_O
            # if ex.error_code != databricks_pb2.RESOURCE_DOES_NOT_EXIST:
            if (
                ex.error_code
                != mlflow.MlflowException(
                    "", databricks_pb2.RESOURCE_DOES_NOT_EXIST
                ).error_code
            ):
                raise

        # Creating MLMD artifact corresponding to MLFlow's RegisteredModel.
        # We use "Artifact" and "system.Model" instead of "VertexModel" since this empty model is not Vertex Model.
        tags_dict = {tag.key: tag.value for tag in tags or []}
        model_artifact = aiplatform.metadata.artifact.Artifact.create(
            schema_title="system.Model",
            # TODO: Set resource_id based on the model name
            # User-specified resource ID must match the regular expression '[a-z0-9][a-z0-9-]{0,127}'
            # resource_id= "google-cloud-mlflow-model-" + sanitized_name + "-" + hexdigest(name)
            display_name=name,
            description=description,
            metadata={
                _MLFLOW_MODEL_KEY: _MLFLOW_MODEL_VALUE,
                _MLFLOW_MODEL_NAME_KEY: name,
                _MLFLOW_MODEL_DESCRIPTION_KEY: description,
                # MLMD supports dict-typed values
                _MLFLOW_MODEL_TAGS_KEY: tags_dict,
                _MLFLOW_MODEL_NEXT_VERSION_KEY: 1,
            },
        )
        registered_mlflow_model = model_registry.RegisteredModel(
            name=name,
            creation_timestamp=int(model_artifact.create_time.timestamp()),
            last_updated_timestamp=int(model_artifact.update_time.timestamp()),
            description=description,
            tags=tags,
        )
        registered_mlflow_model._model_artifact = model_artifact
        return registered_mlflow_model

    def update_registered_model(
        self,
        name: str,
        description: str,
    ) -> model_registry.RegisteredModel:
        """Updates description of the registered model.

        Args:
            name: Registered model name.
            description: New description.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        validation._validate_model_name(name)
        model_artifact = self._get_registered_model_artifact_by_name(name=name)
        model_artifact.update(
            description=description,
            metadata={_MLFLOW_MODEL_DESCRIPTION_KEY: description},
        )
        registered_mlflow_model = GoogleCloudVertexAIModelRegistry._create_registered_model_object_from_model_artifact(
            model_artifact=model_artifact
        )
        return registered_mlflow_model

    def rename_registered_model(
        self,
        name: str,
        new_name: str,
    ) -> model_registry.RegisteredModel:
        """Renames the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        validation._validate_model_name(name)
        validation._validate_model_name(new_name)
        # TODO: ! Handle the model versions
        model_artifact = self._get_registered_model_artifact_by_name(name=name)
        model_artifact.update(metadata={_MLFLOW_MODEL_NAME_KEY: new_name})
        # The client library does not allow changing the artifact display_name
        # model_artifact.display_name = new_name
        registered_mlflow_model = GoogleCloudVertexAIModelRegistry._create_registered_model_object_from_model_artifact(
            model_artifact=model_artifact
        )
        return registered_mlflow_model

    def delete_registered_model(self, name: str) -> None:
        """Deletes the registered model.

        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        validation._validate_model_name(name)
        # TODO: ! Handle the model versions
        model_artifact = self._get_registered_model_artifact_by_name(name=name)
        model_artifact.delete()

    def _list_registered_models(
        self,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> paged_list.PagedList[model_registry.RegisteredModel]:
        """Lists all registered models.

        Args:
            max_results: Maximum number of registered models desired.
            page_token: Token specifying the next page of results. It should be obtained from
                a `list_registered_models` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects.
            The pagination token for the next page can be obtained via the ``token`` attribute
            of the object.
        """
        list_of_model_artifacts = aiplatform.metadata.artifact.Artifact.list(
            filter=f"metadata.{_MLFLOW_MODEL_KEY}.string_value={_MLFLOW_MODEL_VALUE}"
        )
        registered_models = [
            GoogleCloudVertexAIModelRegistry._create_registered_model_object_from_model_artifact(
                model_artifact
            )
            for model_artifact in list_of_model_artifacts
        ]
        if max_results:
            registered_models = registered_models[0:max_results]
        return paged_list.PagedList(items=registered_models, token=None)

    def search_registered_models(
        self,
        filter_string: Optional[str] = None,
        max_results: Optional[int] = None,
        order_by: Optional[str] = None,
        page_token: Optional[str] = None,
    ) -> paged_list.PagedList[model_registry.RegisteredModel]:
        """Searches for registered models that satisfy the filter criteria.

        Args:
            filter_string: Filter query string, defaults to searching all registered models.
            max_results: Maximum number of registered models desired. (Default value = None)
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results. (Default value = None)
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call. (Default value = None)

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.
        """
        del page_token
        parsed_filters = SearchUtils.parse_search_filter(filter_string)
        if order_by:
            (
                ordering_key,
                ordering_is_ascending,
            ) = SearchUtils.parse_order_by_for_search_registered_models(order_by)
        models = self._list_registered_models()
        for parsed_filter in parsed_filters:
            if parsed_filter["comparator"] != "=":
                raise mlflow.exceptions.MlflowException(
                    "Model Registry search filter only supports equality(=) "
                    f"comparator. Input filter string: {filter_string}",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                )
            # Key validated by `parse_filter_for_models`
            key = parsed_filter["key"]
            value = parsed_filter["value"]
            models = [model for model in models if getattr(model, key, None) == value]
        if order_by:
            models.sort(
                key=lambda x: getattr(x, ordering_key, None),
                reversed=not ordering_is_ascending,
            )
        if max_results:
            models = models[0:max_results]
        return models

    def get_registered_model(self, name: str) -> model_registry.RegisteredModel:
        """Gets registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        validation._validate_model_name(name)
        model_artifact = self._get_registered_model_artifact_by_name(name=name)
        registered_mlflow_model = GoogleCloudVertexAIModelRegistry._create_registered_model_object_from_model_artifact(
            model_artifact=model_artifact
        )
        return registered_mlflow_model

    def _get_registered_model_artifact_by_name(self, name: str) -> aiplatform.Model:
        # TODO: Get the model artifact based on the model name-based resource_id
        list_of_model_artifacts = aiplatform.metadata.artifact.Artifact.list(
            filter=f'metadata.{_MLFLOW_MODEL_KEY}.string_value={_MLFLOW_MODEL_VALUE} AND metadata.{_MLFLOW_MODEL_NAME_KEY}.string_value="{name}"'
        )
        if not list_of_model_artifacts:
            raise mlflow.MlflowException(
                f"Registered Model with name={name} not found",
                error_code=databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        if len(list_of_model_artifacts) > 1:
            raise mlflow.MlflowException(
                f"Expected only 1 registered model with name={name}. "
                f"Found: {list_of_model_artifacts}.",
                error_code=databricks_pb2.INVALID_STATE,
            )
        return list_of_model_artifacts[0]

    @staticmethod
    def _create_registered_model_object_from_model_artifact(model_artifact):
        name = model_artifact.metadata[_MLFLOW_MODEL_NAME_KEY]
        description = model_artifact.metadata[_MLFLOW_MODEL_DESCRIPTION_KEY]
        tags_dict = model_artifact.metadata[_MLFLOW_MODEL_TAGS_KEY]
        tags = [
            model_registry.RegisteredModelTag(key=key, value=value)
            for key, value in tags_dict.items()
            if value is not None
        ]
        registered_mlflow_model = model_registry.RegisteredModel(
            name=name,
            creation_timestamp=int(model_artifact.create_time.timestamp()),
            last_updated_timestamp=int(model_artifact.update_time.timestamp()),
            description=description,
            tags=tags,
            # TODO:
            # latest_versions=...
        )
        registered_mlflow_model._model_artifact = model_artifact
        return registered_mlflow_model

    def set_registered_model_tag(
        self,
        name: str,
        tag: model_registry.RegisteredModelTag,
    ) -> None:
        """Sets a tag for the registered model.

        Args:
            name: Registered model name.
            tag: py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        validation._validate_model_name(name)
        validation._validate_registered_model_tag(tag.key, tag.value)

        model_artifact = self._get_registered_model_artifact_by_name(name=name)
        tags_dict = model_artifact.metadata[_MLFLOW_MODEL_TAGS_KEY]
        tags_dict[tag.key] = tag.value
        model_artifact.update(metadata={_MLFLOW_MODEL_TAGS_KEY: tags_dict})
        registered_mlflow_model = GoogleCloudVertexAIModelRegistry._create_registered_model_object_from_model_artifact(
            model_artifact=model_artifact
        )
        return registered_mlflow_model

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        validation._validate_model_name(name)
        validation._validate_tag_name(key)
        model_artifact = self._get_registered_model_artifact_by_name(name=name)
        tags_dict = model_artifact.metadata[_MLFLOW_MODEL_TAGS_KEY]
        # The Vertex SDK Artifact metadata interface does not seem to allow deleting keys or sub-keys in the metadata
        # Workaround is to set the tag to None
        # del tags_dict[key]
        if key in tags_dict:
            tags_dict[key] = None
        model_artifact.update(metadata={_MLFLOW_MODEL_TAGS_KEY: tags_dict})
        registered_mlflow_model = GoogleCloudVertexAIModelRegistry._create_registered_model_object_from_model_artifact(
            model_artifact=model_artifact
        )
        return registered_mlflow_model

    def get_latest_versions(
        self, name: str, stages: List[str] = None
    ) -> List[model_registry.ModelVersion]:
        """Gets the latest model version for each requested stage.

        If no ``stages`` argument is provided, returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                for 'Staging' and 'Production' stages. (Default value = None)

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        validation._validate_model_name(name)
        stages = stages or model_version_stages.DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        stages = [model_version_stages.get_canonical_stage(stage) for stage in stages]
        latest_versions = {}
        model_versions = self._list_model_versions(name=name)
        for model_version in model_versions:
            stage = model_version.current_stage
            if stage not in stages:
                continue
            if stage not in latest_versions or int(
                latest_versions[stage].version
            ) < int(model_version.version):
                latest_versions[stage] = model_version
        latest_versions_list = [latest_versions.get(stage) or None for stage in stages]
        return latest_versions_list

    # CRUD API for ModelVersion objects

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: str = None,
        tags: List[model_registry.ModelVersionTag] = None,
        run_link: str = None,
        description: str = None,
    ) -> model_registry.ModelVersion:
        """Creates a new model version from given source and run ID.

        Args:
            name: Registered model name.
            source: Source path where the MLflow model is stored.
            run_id: Run ID from MLflow tracking server that generated the model. (Default value = None)
            tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                instances associated with this model version. (Default value = None)
            run_link: Link to the run from an MLflow tracking server that generated this model. (Default value = None)
            description: Description of the version. (Default value = None)

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.
        """
        validation._validate_model_name(name)
        for tag in tags or []:
            validation._validate_model_version_tag(tag.key, tag.value)

        registered_model_artifact = self._get_registered_model_artifact_by_name(
            name=name
        )

        # TODO: ! Avoid race conditions when incrementing version
        version = registered_model_artifact.metadata[_MLFLOW_MODEL_NEXT_VERSION_KEY]
        next_version = version + 1
        registered_model_artifact.update(
            metadata={_MLFLOW_MODEL_NEXT_VERSION_KEY: next_version}
        )

        # Creating MLMD artifact corresponding to MLFlow's ModelVersion.
        # For now we use "Artifact" and "system.Model" instead of "VertexModel" since we do not upload the model to Vertex Model Registry yet.
        tags_dict = {tag.key: tag.value for tag in tags or []}
        model_version_artifact = aiplatform.metadata.artifact.Artifact.create(
            schema_title="system.Model",
            # TODO: Set resource_id based on the model name
            # User-specified resource ID must match the regular expression '[a-z0-9][a-z0-9-]{0,127}'
            # resource_id= "google-cloud-mlflow-model-" + sanitized_name + "-" + hexdigest(name)
            display_name=name,
            description=description,
            metadata={
                _MLFLOW_MODEL_VERSION_KEY: _MLFLOW_MODEL_VERSION_VALUE,
                _MLFLOW_MODEL_VERSION_NAME_KEY: name,
                _MLFLOW_MODEL_VERSION_PARENT_MODEL_ARTIFACT_NAME_KEY: registered_model_artifact.resource_name,
                _MLFLOW_MODEL_VERSION_VERSION_KEY: version,
                _MLFLOW_MODEL_VERSION_STAGE_KEY: None,
                _MLFLOW_MODEL_VERSION_DESCRIPTION_KEY: description,
                _MLFLOW_MODEL_VERSION_SOURCE_KEY: source,
                _MLFLOW_MODEL_VERSION_RUN_ID_KEY: run_id,
                _MLFLOW_MODEL_VERSION_RUN_LINK_KEY: run_link,
                # MLMD supports dict-typed values
                _MLFLOW_MODEL_VERSION_TAGS_KEY: tags_dict,
                # _MLFLOW_MODEL_VERSION_OBJECT_KEY: _dump_registered_model_version_to_dict(model_version),
            },
        )
        model_version = GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
            model_version_artifact
        )
        return model_version

    @staticmethod
    def _create_model_version_object_from_model_artifact(
        model_artifact: aiplatform.metadata.artifact.Artifact,
    ) -> model_registry.ModelVersion:
        tags_dict = model_artifact.metadata[_MLFLOW_MODEL_VERSION_TAGS_KEY]
        tags = [
            model_registry.RegisteredModelTag(key=key, value=value)
            for key, value in tags_dict.items()
            if value is not None
        ]
        model_version = model_registry.ModelVersion(
            name=model_artifact.metadata[_MLFLOW_MODEL_VERSION_NAME_KEY],
            version=model_artifact.metadata[_MLFLOW_MODEL_VERSION_VERSION_KEY],
            creation_timestamp=int(model_artifact.create_time.timestamp()),
            last_updated_timestamp=int(model_artifact.update_time.timestamp()),
            description=model_artifact.metadata[_MLFLOW_MODEL_VERSION_DESCRIPTION_KEY],
            # user_id=None,
            current_stage=model_artifact.metadata[_MLFLOW_MODEL_VERSION_STAGE_KEY],
            source=model_artifact.metadata[_MLFLOW_MODEL_VERSION_SOURCE_KEY],
            run_id=model_artifact.metadata[_MLFLOW_MODEL_VERSION_RUN_ID_KEY],
            # status='READY',
            # status_message=None,
            tags=tags,
            run_link=model_artifact.metadata[_MLFLOW_MODEL_VERSION_RUN_LINK_KEY],
        )
        model_version._model_artifact = model_artifact
        return model_version

    def get_model_version(self, name: str, version: str) -> model_registry.ModelVersion:
        """Gets the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        model_artifact = self._get_model_version_artifact(name=name, version=version)
        model_version = GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
            model_artifact
        )
        return model_version

    def _get_model_version_artifact(
        self, name: str, version: str
    ) -> aiplatform.metadata.artifact.Artifact:
        """Gets the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        validation._validate_model_name(name)
        validation._validate_model_version(version)

        list_of_model_artifacts = aiplatform.metadata.artifact.Artifact.list(
            filter=f'metadata.{_MLFLOW_MODEL_VERSION_NAME_KEY}.string_value="{name}" AND metadata.{_MLFLOW_MODEL_VERSION_VERSION_KEY}.number_value="{int(version)}"'
        )
        if not list_of_model_artifacts:
            raise mlflow.MlflowException(
                f"Registered Model with name={name} version={version} not found",
                error_code=databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        if len(list_of_model_artifacts) > 1:
            raise mlflow.MlflowException(
                f"Expected only 1 registered model with name={name}  version={version}. "
                f"Found: {list_of_model_artifacts}.",
                error_code=databricks_pb2.INVALID_STATE,
            )
        return list_of_model_artifacts[0]

    def update_model_version(
        self,
        name: str,
        version: str,
        description: str,
    ) -> model_registry.ModelVersion:
        """Updates metadata associated with a model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        model_artifact = self._get_model_version_artifact(name=name, version=version)
        model_artifact.update(
            description=description,
            metadata={_MLFLOW_MODEL_VERSION_DESCRIPTION_KEY: description},
        )
        model_version = GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
            model_artifact
        )
        return model_version

    def delete_model_version(self, name: str, version: str) -> None:
        """Deletes model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """
        model_artifact = self._get_model_version_artifact(name=name, version=version)
        model_artifact.delete()

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """Gets the download location in Model Registry for this model version.

        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single URI location that allows reads for downloading.
        """
        model_version = self.get_model_version(name, version)
        return model_version.source

    def set_model_version_tag(
        self, name: str, version: str, tag: model_registry.ModelVersionTag
    ) -> None:
        """Sets a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: py:class:`mlflow.entities.model_registry.ModelVersionTag` instance
                to log.

        Returns:
            None
        """
        validation._validate_model_version_tag(tag.key, tag.value)
        model_artifact = self._get_model_version_artifact(name, version)

        tags_dict = model_artifact.metadata[_MLFLOW_MODEL_VERSION_TAGS_KEY]
        tags_dict[tag.key] = tag.value
        model_artifact.update(metadata={_MLFLOW_MODEL_VERSION_TAGS_KEY: tags_dict})
        model_version = GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
            model_artifact
        )
        return model_version

    def delete_model_version_tag(self, name: str, version: str, key: str) -> None:
        """Deletes a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        validation._validate_tag_name(key)
        model_version = self.get_model_version(name, version)
        model_artifact = self._get_model_version_artifact(name, version)

        tags_dict = model_artifact.metadata[_MLFLOW_MODEL_VERSION_TAGS_KEY]
        # The Vertex SDK Artifact metadata interface does not seem to allow deleting keys or sub-keys in the metadata
        # Workaround is to set the tag to None
        # del tags_dict[key]
        if key in tags_dict:
            tags_dict[key] = None
        model_artifact.update(metadata={_MLFLOW_MODEL_VERSION_TAGS_KEY: tags_dict})
        model_version = GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
            model_artifact
        )
        return model_version

    def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False,
    ) -> model_registry.ModelVersion:
        """Updates model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only valid
                when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        Returns:
              A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        stage = model_version_stages.get_canonical_stage(stage)
        model_artifact = self._get_model_version_artifact(name, version)
        model_artifact.update(metadata={_MLFLOW_MODEL_VERSION_STAGE_KEY: stage})
        model_version = GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
            model_artifact
        )

        if archive_existing_versions:
            for other_model_version in self._list_model_versions(name=name):
                # ! Cannot compare `other_model_version.version` with `version` since they have different types
                if (
                    other_model_version.name == name
                    and other_model_version.version != model_version.version
                    and other_model_version.current_stage == stage
                ):
                    print(f"Setting={other_model_version}")
                    other_model_artifact = other_model_version._model_artifact
                    other_model_artifact.update(
                        metadata={
                            _MLFLOW_MODEL_VERSION_STAGE_KEY: model_version_stages.STAGE_ARCHIVED
                        }
                    )

        return model_version

    def _list_model_versions(
        self,
        name: Optional[str] = None,
    ) -> List[model_registry.ModelVersion]:
        """Lists all versions of a registered model.

        Args:
            name: Registered model name.

        Returns:
            A list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        if name:
            validation._validate_model_name(name)
            list_of_model_artifacts = aiplatform.metadata.artifact.Artifact.list(
                filter=f'metadata.{_MLFLOW_MODEL_VERSION_NAME_KEY}.string_value="{name}"'
            )
        else:
            list_of_model_artifacts = aiplatform.metadata.artifact.Artifact.list(
                filter=f'metadata.{_MLFLOW_MODEL_VERSION_KEY}.string_value="{_MLFLOW_MODEL_VERSION_VALUE}"'
            )

        models_versions = [
            GoogleCloudVertexAIModelRegistry._create_model_version_object_from_model_artifact(
                model_artifact
            )
            for model_artifact in list_of_model_artifacts
        ]
        return models_versions

    def search_model_versions(
        self,
        filter_string: Optional[str] = None,
    ) -> paged_list.PagedList[model_registry.ModelVersion]:
        """Searches for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'`` or ``run_id = '...'``.

        Returns:
            PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
            objects.
        """
        parsed_filters = SearchUtils.parse_search_filter(filter_string)
        model_versions = self._list_model_versions()
        model_versions = [
            model_version
            for model_version in model_versions
            if model_version.current_stage is None
            or model_version.current_stage in model_version_stages.ALL_STAGES
        ]
        for parsed_filter in parsed_filters:
            if parsed_filter["comparator"] != "=":
                raise mlflow.exceptions.MlflowException(
                    "Model Registry search filter only supports equality(=) "
                    f"comparator. Input filter string: {filter_string}",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                )
            # Key validated by `parse_filter_for_model_versions`
            key = parsed_filter["key"]
            value = parsed_filter["value"]
            model_versions = [
                model_version
                for model_version in model_versions
                if getattr(model_version, key, None) == value
            ]
        return paged_list.PagedList(items=model_versions, token=None)
