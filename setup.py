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
"""MLflow Google Cloud Vertex AI integration package."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="google_cloud_mlflow",
    version="0.0.6",
    description="MLflow Google Cloud Vertex AI integration package",
    url="https://github.com/Ark-kun/google_cloud_mlflow",
    project_urls={
        'Source': 'https://github.com/pypa/sampleproject/',
        'Issues': 'https://github.com/Ark-kun/google_cloud_mlflow/issues',
    },
    author="Alexey Volkov",
    author_email="alexey.volkov@ark-kun.com",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    license="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    keywords='mlflow, Google Cloud, Vertex AI',
    python_requires=">=3.6",
    install_requires=[
        "google-cloud-aiplatform~=1.7",
        "mlflow~=1.22",
        "google-cloud-build==3.*",
        "google-cloud-storage==1.*",
    ],
    entry_points={
        "mlflow.deployments": [
            "google_cloud=google_cloud_mlflow.mlflow_model_deployment_plugin_for_google_cloud_vertex_ai",
        ],
        "mlflow.model_registry_store": [
            "gs=google_cloud_mlflow.mlflow_model_registry_plugin_for_google_cloud_storage:GoogleCloudStorageModelRegistry",
        ],
    },
)
