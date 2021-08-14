"""MLflow Model Registry plugin to allow logging models to Google Cloud Storage."""

import datetime
from typing import Iterator, List

import google
from google.cloud import storage
from google.protobuf import json_format
import mlflow
from mlflow.entities import model_registry
from mlflow.entities.model_registry import model_version_stages
from mlflow.protos import databricks_pb2
from mlflow.store.entities import paged_list
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.validation import (
    _validate_registered_model_tag,
    _validate_model_version_tag,
    _validate_model_name,
    _validate_model_version,
    _validate_tag_name,
)

MODEL_INFO_FILE_NAME: str = "model_info.json"
MODEL_VERSION_INFO_FILE_NAME: str = "model_version_info.json"
LAST_MODEL_VERSION_FILE_NAME: str = "last_model_version"

REALLY_DELETE_MODEL_VERSIONS = False


class GoogleCloudStorageModelRegistry(
    mlflow.store.model_registry.abstract_store.AbstractStore
):
    """Class for storing Model Registry metadata."""

    def __init__(self, base_uri: str):
        if not _validate_base_uri(base_uri):
            raise mlflow.MlflowException(f"Bad base_uri format: {base_uri}")
        base_uri = base_uri.rstrip("/") + "/"
        self._base_uri = base_uri
        # bucket, path = base_uri[len("gs://") :].split("/", 1)
        # self._bucket = bucket
        # self._path_prefix = path

    # CRUD API for RegisteredModel objects

    def _get_model_dir(self, name: str) -> str:
        # return self._path_prefix.rstrip("/") + "/" + name + "/"
        return self._base_uri + name + "/"

    def _get_model_info_file_path(self, name: str) -> str:
        return self._get_model_dir(name=name) + MODEL_INFO_FILE_NAME

    def _get_model_version_dir(self, name: str, version: str) -> str:
        return self._get_model_dir(name=name) + version + "/"

    def _get_model_version_info_file_path(self, name: str, version: str) -> str:
        return (
            self._get_model_version_dir(name=name, version=version)
            + MODEL_VERSION_INFO_FILE_NAME
        )

    def _get_model_blob_path(self, name: str, version: str) -> str:
        return f"{self._path_prefix}/{name}/{version}/"

    def create_registered_model(
        self,
        name: str,
        tags: List[model_registry.RegisteredModelTag] = None,
        description: str = None,
    ) -> model_registry.RegisteredModel:
        """Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.
        """
        # TODO(avolkov): Validate that the model does nto exist
        _validate_model_name(name)
        for tag in tags or []:
            _validate_registered_model_tag(tag.key, tag.value)
        current_time = datetime.datetime.utcnow()
        model = model_registry.RegisteredModel(
            name=name,
            creation_timestamp=current_time.timestamp,
            last_updated_timestamp=current_time.timestamp,
            description=description,
            tags=tags,
        )
        model_json = json_format.MessageToJson(model.to_proto())
        model_uri = self._get_model_info_file_path(name=name)
        storage.Blob.from_string(uri=model_uri).upload_from_string(data=model_json)
        return model

    def update_registered_model(
        self,
        name: str,
        description: str,
    ) -> model_registry.RegisteredModel:
        """Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        model = self.get_registered_model(name=name)
        model.description = description
        self._set_registered_model(name, model)
        return model

    def rename_registered_model(
        self,
        name: str,
        new_name: str,
    ) -> model_registry.RegisteredModel:
        """Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)
        _validate_model_name(new_name)
        src_dir = self._get_model_dir(name=name)
        dst_dir = self._get_model_dir(name=new_name)
        src_dir_blob = storage.Blob.from_string(uri=src_dir)
        dst_dir_blob = storage.Blob.from_string(uri=dst_dir)
        bucket = src_dir_blob.bucket
        src_path = src_dir_blob.path
        dst_path = dst_dir_blob.path
        blobs: Iterator[storage.Blob] = storage.Client().list_blobs(
            bucket_or_name=bucket,
            prefix=src_path,
        )
        for blob in blobs:
            assert blob.path.startswith(src_path)
            new_path = blob.path.replace(src_path, dst_path, 1)
            blob.bucket.rename_blob(blob, new_path)

    def delete_registered_model(self, name: str) -> None:
        """Delete the registered model.

        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        _validate_model_name(name)
        src_dir = self._get_model_dir(name=name)
        src_dir_blob = storage.Blob.from_string(uri=src_dir)
        bucket = src_dir_blob.bucket
        src_path = src_dir_blob.path
        blobs: Iterator[storage.Blob] = storage.Client().list_blobs(
            bucket_or_name=bucket,
            prefix=src_path,
        )
        bucket.delete_blobs(blobs)

    def list_registered_models(
        self, max_results: int, page_token: str
    ) -> paged_list.PagedList[model_registry.RegisteredModel]:
        """List of all registered models.

        Args:
            max_results: Maximum number of registered models desired.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``list_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects.
            The pagination token for the next page can be obtained via the ``token`` attribute
            of the object.
        """
        root_dir = storage.Blob.from_string(uri=self._base_uri)
        blob_iterator = storage.Client().list_blobs(
            bucket_or_name=root_dir.bucket.name,
            prefix=root_dir.path,
            max_results=max_results,
            page_token=page_token,
        )
        models = [
            _json_to_registered_model(blob.download_as_text())
            for blob in blob_iterator
            if blob.path.endswith(MODEL_INFO_FILE_NAME)
        ]
        return paged_list.PagedList(items=models, token=blob_iterator.next_page_token)

    def search_registered_models(
        self,
        filter_string: str = None,
        max_results: int = None,
        order_by: str = None,
        page_token: str = None,
    ) -> paged_list.PagedList[model_registry.RegisteredModel]:
        """Search for registered models in backend that satisfy the filter criteria.

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
        parsed_filters = SearchUtils.parse_filter_for_model_versions(filter_string)
        (
            ordering_key,
            ordering_is_ascending,
        ) = SearchUtils.parse_order_by_for_search_registered_models(order_by)
        model_versions = self._list_model_versions(name=...)
        model_versions = [
            model_version
            for model_version in model_versions
            if model_version.current_stage is None
            or model_version.current_stage in model_version_stages.ALL_STAGES
        ]
        for parsed_filter in parsed_filters:
            if parsed_filter["comparator"] != "=":
                raise mlflow.MlflowException(
                    f"Model Registry search filter only supports equality(=) "
                    "comparator. Input filter string: {filter_string}",
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
        model_versions.sort(
            key=lambda x: getattr(x, ordering_key, None),
            reversed=not ordering_is_ascending,
        )
        model_versions = model_versions[0:max_results]
        return model_versions

    def get_registered_model(self, name: str) -> model_registry.RegisteredModel:
        """Get registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)
        model_uri = self._get_model_info_file_path(name=name)
        try:
            model_json = storage.Blob.from_string(uri=model_uri).download_as_text()
            model = _json_to_registered_model(model_json)
            return model
        except google.api_core.exceptions.NotFound:
            raise mlflow.MlflowException(
                message=f'Model "{name}" does not exist',
                error_code=databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )

    def _set_registered_model(
        self,
        name: str,
        model: model_registry.RegisteredModel,
        update_modification_time: bool = True,
    ) -> None:
        """Set registered model instance.

        Args:
            name: Registered model name.
            model: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
            update_modification_time: Whether to update the modification time
        """
        _validate_model_name(name)
        if update_modification_time:
            current_time = datetime.datetime.utcnow()
            model.last_updated_timestamp = current_time.timestamp
        model_json = json_format.MessageToJson(model.to_proto())
        model_uri = self._get_model_info_file_path(name=name)
        storage.Blob.from_string(uri=model_uri).upload_from_string(data=model_json)

    def get_latest_versions(
        self, name: str, stages: List[str] = None
    ) -> List[model_registry.ModelVersion]:
        """Latest version models for each requested stage.

        If no ``stages`` argument is provided, returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                for 'Staging' and 'Production' stages. (Default value = None)

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        _validate_model_name(name)
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
        latest_versions_list = [latest_versions[stage] or None for stage in stages]
        raise latest_versions_list

    def set_registered_model_tag(
        self,
        name: str,
        tag: model_registry.RegisteredModelTag,
    ) -> None:
        """Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        model = self.get_registered_model(name=name)
        model.tags[tag.key] = tag.value
        self._set_registered_model(name, model)

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        model = self.get_registered_model(name=name)
        del model.tags[key]
        self._set_registered_model(name, model)

    # CRUD API for ModelVersion objects

    def _increment_last_model_version(self, name: str) -> int:
        """Increments and returns the last model version for a given model.

        Args:
            name: Registered model name.

        Returns:
            The version number for the next model version.
        """
        _validate_model_name(name)
        model_dir_uri = self._get_model_dir(name=name)
        last_model_version_file_uri = model_dir_uri + LAST_MODEL_VERSION_FILE_NAME
        last_model_version_file_blob = storage.Blob.from_string(
            uri=last_model_version_file_uri
        )
        last_model_version = 0
        try:
            last_model_version = int(last_model_version_file_blob.download_as_text())
        except google.api_core.exceptions.NotFound:
            pass
        last_model_version += 1
        last_model_version_file_blob.upload_from_string(
            data=str(last_model_version),
            # Avoiding race condition
            if_generation_match=last_model_version_file_blob.generation,
        )
        return last_model_version

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: str = None,
        tags: List[model_registry.ModelVersionTag] = None,
        run_link: str = None,
        description: str = None,
    ) -> model_registry.ModelVersion:
        """Create a new model version from given source and run ID.

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
        _validate_model_name(name)
        version = self._increment_last_model_version(name=name)
        current_time = datetime.datetime.utcnow()
        model_version = model_registry.ModelVersion(
            name=name,
            version=version,
            creation_timestamp=current_time.timestamp,
            last_updated_timestamp=current_time.timestamp,
            description=description,
            source=source,
            run_id=run_id,
            tags=tags,
            run_link=run_link,
        )
        self._set_model_version(name=name, version=version, model_version=model_version)

    def update_model_version(
        self, name: str, version: str, description: str,
    ) -> model_registry.ModelVersion:
        """Update metadata associated with a model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        model_version = self.get_model_version(name=name, version=version)
        model_version.description = description
        self._set_model_version(name=name, version=version, model_version=model_version)

    def transition_model_version_stage(
        self, name: str, version: str, stage: str, archive_existing_versions: bool
    ) -> model_registry.ModelVersion:
        """Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automically moved to the "archived" stage. Only valid
                when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        Returns:
              A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        model_version = self.get_model_version(name=name, version=version)
        stage = model_version_stages.get_canonical_stage(stage)
        model_version.current_stage = stage
        self._set_model_version(name, version, model_version)
        if archive_existing_versions:
            all_model_versions = self._list_model_versions(name=name)
            all_other_model_versions = list(
                filter(
                    lambda model_version: model_version.name == name, all_model_versions
                )
            )
            all_other_model_versions_in_same_stage = list(
                filter(
                    lambda model_version: model_version.current_stage == stage,
                    all_other_model_versions,
                )
            )
            for other_model_version in all_other_model_versions_in_same_stage:
                other_model_version.current_stage = model_version_stages.STAGE_ARCHIVED
                self._set_model_version(
                    name=other_model_version.name,
                    version=other_model_version.version,
                    model_version=other_model_version,
                )

    def delete_model_version(self, name: str, version: str) -> None:
        """Delete model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """
        # Validating that the version is a proper model version
        model_version = self.get_model_version(name=name, version=version)
        if REALLY_DELETE_MODEL_VERSIONS:
            model_version_uri = self._get_model_version_info_file_path(
                name=name, version=version
            )
            storage.Blob.from_string(uri=model_version_uri).delete()
        else:
            model_version.current_stage = model_version_stages.STAGE_DELETED_INTERNAL
            self._set_model_version(
                name=name,
                version=version,
                model_version=model_version,
            )

    def get_model_version(self, name: str, version: str) -> model_registry.ModelVersion:
        """Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        model_version_uri = self._get_model_version_info_file_path(
            name=name, version=version
        )
        try:
            model_version_json = storage.Blob.from_string(
                uri=model_version_uri
            ).download_as_text()
            model_version = _json_to_registered_model_version(model_version_json)
            return model_version
        except google.api_core.exceptions.NotFound:
            raise mlflow.MlflowException(
                message=f'Model "{name}" version "{version}" does not exist',
                error_code=databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )

    def _set_model_version(
        self,
        name: str,
        version: str,
        model_version: model_registry.ModelVersion,
        update_modification_time: bool = True,
    ) -> None:
        """Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.
            model_version: A :py:class:`mlflow.entities.model_registry.ModelVersion` object.
            update_modification_time: Whether to update the modification time

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_model_version(version)
        if update_modification_time:
            current_time = datetime.datetime.utcnow()
            model_version.last_updated_timestamp = current_time.timestamp
        model_version_json = json_format.MessageToJson(model_version.to_proto())
        model_version_uri = self._get_model_version_info_file_path(
            name=name, version=version
        )
        storage.Blob.from_string(uri=model_version_uri).upload_from_string(
            model_version_json
        )

    def _list_model_versions(self, name: str) -> List[model_registry.ModelVersion]:
        """List of all versions of a registered model.

        Args:
            name: Registered model name.

        Returns:
            A list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        _validate_model_name(name)
        model_uri = self._get_model_dir(name=name)
        model_dir_blob = storage.Blob.from_string(uri=model_uri)
        blob_iterator = storage.Client().list_blobs(
            bucket_or_name=model_dir_blob.bucket,
            prefix=model_dir_blob.path,
        )
        models = [
            _json_to_registered_model_version(blob.download_as_text())
            for blob in blob_iterator
            if blob.path.endswith(MODEL_VERSION_INFO_FILE_NAME)
        ]
        return models

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """Get the download location in Model Registry for this model version.

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

    def search_model_versions(
        self, filter_string: str
    ) -> paged_list.PagedList[model_registry.ModelVersion]:
        """Search for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'`` or ``run_id = '...'``.

        Returns:
            PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
            objects.
        """
        parsed_filters = SearchUtils.parse_filter_for_model_versions(filter_string)
        model_versions = self._list_model_versions(name=...)
        model_versions = [
            model_version
            for model_version in model_versions
            if model_version.current_stage is None
            or model_version.current_stage in model_version_stages.ALL_STAGES
        ]
        for parsed_filter in parsed_filters:
            if parsed_filter["comparator"] != "=":
                raise mlflow.MlflowException(
                    f"Model Registry search filter only supports equality(=) "
                    "comparator. Input filter string: {filter_string}",
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
        return paged_list.PagedList(model_versions)

    def set_model_version_tag(
        self, name: str, version: str, tag: model_registry.ModelVersionTag
    ) -> None:
        """Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.

        Returns:
            None
        """
        _validate_model_version_tag(tag.key, tag.value)
        model_version = self.get_model_version(name, version)
        model_version.tags[tag.key] = tag.value
        self._set_model_version(name=name, version=version, model_version=model_version)

    def delete_model_version_tag(self, name: str, version: str, key: str) -> None:
        """Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        _validate_tag_name(key)
        model_version = self.get_model_version(name, version)
        del model_version.tags[key]
        self._set_model_version(name=name, version=version, model_version=model_version)


def _validate_base_uri(base_uri: str) -> bool:
    return base_uri.startswith("gs://")


def _json_to_registered_model(model_json: str) -> model_registry.RegisteredModel:
    model = model_registry.RegisteredModel()
    json_format.Parse(model_json, model)
    return model


def _json_to_registered_model_version(model_json: str) -> model_registry.ModelVersion:
    model = model_registry.ModelVersion()
    json_format.Parse(model_json, model)
    return model
