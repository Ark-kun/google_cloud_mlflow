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
import logging
import subprocess
import shutil
import tempfile
import uuid


_logger = logging.getLogger(__name__)


def _build_image_from_context(context_dir: str, image_name: str):
    return _build_image_from_context_using_cloudbuild_client(
        context_dir=context_dir,
        image_name=image_name,
    )


def _build_image_from_context_using_cloudbuild_gcloud(
    context_dir: str,
    image_name: str,
):
    build_process = subprocess.Popen(
        args=["gcloud", "builds", "submit", "--tag", image_name, "--timeout", "1800", context_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    info_line_prefixes = [
        "Logs are available at ",
        "Step ",
        "Pushing ",
        "latest: digest: ",
    ]
    for line in iter(build_process.stdout.readline, ""):
        line = line.rstrip()
        if any(line.startswith(prefix) for prefix in info_line_prefixes):
            _logger.info(line)
        else:
            _logger.debug(line)

    if build_process.wait() != 0:
        raise RuntimeError("Container image build has failed.")
    return


def _build_image_from_context_using_cloudbuild_client(
    context_dir: str, image_name: str
):
    import google
    import google.auth
    from google.cloud import storage
    from google.cloud.devtools import cloudbuild

    archive_base_name = tempfile.mktemp()
    context_archive_path = shutil.make_archive(
        base_name=archive_base_name,
        format="gztar",
        root_dir=context_dir,
    )

    _, project_id = google.auth.default()

    storage_client = storage.Client(project=project_id)
    build_client = cloudbuild.CloudBuildClient()

    # Staging the data in GCS
    bucket_name = project_id + "_cloudbuild"

    bucket = storage_client.lookup_bucket(bucket_name)
    # TODO: Throw error if bucket is in different project
    if bucket is None:
        bucket = storage_client.create_bucket(bucket_name)
    blob_name = f"source/{uuid.uuid4().hex}.tgz"

    bucket.blob(blob_name).upload_from_filename(context_archive_path)

    build_config = cloudbuild.Build(
        source=cloudbuild.Source(
            storage_source=cloudbuild.StorageSource(
                bucket=bucket_name, object_=blob_name
            ),
        ),
        images=[image_name],
        steps=[
            cloudbuild.BuildStep(
                name="gcr.io/cloud-builders/docker",
                args=[
                    "build",
                    "--network",
                    "cloudbuild",
                    "--no-cache",
                    "-t",
                    image_name,
                    ".",
                ],
            ),
        ],
        timeout=google.protobuf.duration_pb2.Duration(
            seconds=1800,
        ),
    )
    build_operation = build_client.create_build(
        project_id=project_id, build=build_config
    )
    _logger.info("Submitted Cloud Build job")
    _logger.debug("build_operation.metadata:")
    _logger.debug(build_operation.metadata)
    _logger.info(f"Logs are available at [{build_operation.metadata.build.log_url}].")

    try:
        result = build_operation.result()
    except Exception as ex:
        _logger.error(
            "MLFlow container image build has failed."
            f" See Google Cloud Build logs here: {build_operation.metadata.build.log_url}"
        )
        raise Exception("MLFlow container image build has failed.") from ex
    _logger.debug("operation.result")
    _logger.debug(result)

    built_image = result.results.images[0]
    image_base_name = built_image.name.split(":")[0]
    image_digest = built_image.digest
    image_name_with_digest = image_base_name + "@" + image_digest
    return image_name_with_digest
