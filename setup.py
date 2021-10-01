"""MLflow Google Cloud Vertex AI integration package."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="google_cloud_mlflow",
    version="0.0.4.1",
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
        "google-cloud-aiplatform>=1.3.0",
        "mlflow~=1.20",
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
