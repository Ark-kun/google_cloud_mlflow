import os
import subprocess
import logging

from mlflow.utils.file_utils import TempDir

from mlflow.models import docker_utils

_logger = logging.getLogger(__name__)

DISABLE_ENV_CREATION = "MLFLOW_DISABLE_ENV_CREATION"

_DOCKERFILE_TEMPLATE = docker_utils._DOCKERFILE_TEMPLATE

def _build_image(image_name, entrypoint, mlflow_home=None, custom_setup_steps_hook=None):
    """
    Build an MLflow Docker image that can be used to serve a
    The image is built locally and it requires Docker to run.

    :param image_name: Docker image name.
    :param entry_point: String containing ENTRYPOINT directive for docker image
    :param mlflow_home: (Optional) Path to a local copy of the MLflow GitHub repository.
                        If specified, the image will install MLflow from this directory.
                        If None, it will install MLflow from pip.
    :param custom_setup_steps_hook: (Optional) Single-argument function that takes the string path
           of a dockerfile context directory and returns a string containing Dockerfile commands to
           run during the image build step.
    """

    mlflow_home = os.path.abspath(mlflow_home) if mlflow_home else None
    with TempDir() as tmp:
        cwd = tmp.path()
        install_mlflow = docker_utils._get_mlflow_install_step(cwd, mlflow_home)
        custom_setup_steps = custom_setup_steps_hook(cwd) if custom_setup_steps_hook else ""
        with open(os.path.join(cwd, "Dockerfile"), "w") as f:
            f.write(
                _DOCKERFILE_TEMPLATE.format(
                    install_mlflow=install_mlflow,
                    custom_setup_steps=custom_setup_steps,
                    entrypoint=entrypoint,
                )
            )
        _logger.info("Building docker image with name %s", image_name)
        _build_image_from_context(context_dir=cwd, image_name=image_name)

def _build_image_from_context(context_dir: str, image_name: str):
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
    import google.auth
    from google.cloud import storage
    from google.cloud.devtools import cloudbuild
    import uuid

    _, project_id = google.auth.default()

    storage_client = storage.Client(project=project_id)
    build_client = cloudbuild.CloudBuildClient()

    # TODO: Need to stage the data in GCS
    bucket_name = project_id + "_cloudbuild"
    
    bucket = storage_client.lookup_bucket(bucket_name)
    # TODO: Throw error if bucket is in different project
    if bucket is None:
        bucket = storage_client.create_bucket(bucket_name)
    blob_name = f"source/{uuid.uuid4().hex}.tgz"

    context_archive_path = ...

    # !!! TODO: Archive

    bucket.blob(blob_name).upload_from_filename(context_archive_path)

    build_config = cloudbuild.Build(
        source=cloudbuild.Source(
            storage_source=cloudbuild.StorageSource(
                bucket=bucket_name,
                object=blob_name
            ),
        ),
        images=[image_name],
        steps=[
            cloudbuild.BuildStep(
                name='gcr.io/cloud-builders/docker',
                args=[
                    'build', '--network', 'cloudbuild', '--no-cache', '-t',
                    image_name, '.'
                ],
            ),
        ],
    )
    build_client.create_build(project_id=project_id, build=cloudbuild.Build.b)
    # trigger.build = messages.Build(steps=[
    #     messages.BuildStep(
    #         name='gcr.io/cloud-builders/docker',
    #         dir=args.dockerfile_dir,
    #         args=['build', '-t', image, '-f', args.dockerfile, '.'],
    #     )
    # ])

