# MLflow plugin for Google Cloud Vertex AI

Note: The plugin is **experimental** and may be changed or removed in the future.

## Installation

```shell
python3 -m pip install google_cloud_mlflow
```

## Deployment plugin usage

### Command-line

Create deployment

```shell
mlflow deployments create --target google_cloud --name "deployment name" --model-uri "models:/mymodel/mymodelversion" --config destination_image_uri="gcr.io/<repo>/<path>"
```

List deployments

```shell
mlflow deployments list --target google_cloud
```

Get deployment

```shell
mlflow deployments get --target google_cloud --name "deployment name"
```

Delete deployment

```shell
mlflow deployments delete --target google_cloud --name "deployment name"
```

Update deployment

```shell
mlflow deployments update --target google_cloud --name "deployment name" --model-uri "models:/mymodel/mymodelversion" --config destination_image_uri="gcr.io/<repo>/<path>"
```

Predict

```shell
mlflow deployments predict --target google_cloud --name "deployment name" --input-path "inputs.json" --output-path "outputs.json
```

Get help

```shell
mlflow deployments help --target google_cloud
```

### Python

```python
from mlflow import deployments
client = deployments.get_deploy_client("google_cloud")

# Create deployment
model_uri = "models:/mymodel/mymodelversion"
deployment = client.create_deployment(
    name="deployment name",
    model_uri=model_uri,
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

        # Endpoint config
        endpoint_description=None,
        endpoint_deploy_timeout=None,

        # Vertex AI config
        project=None,
        location=None,
        encryption_spec_key_name=None,
        staging_bucket=None,
    )
)

# List deployments
deployments = client.list_deployments()

# Get deployment
deployments = client.get_deployment(name="deployment name")

# Delete deployment
deployment = client.delete_deployment(name="deployment name")

# Update deployment
deployment = client.create_deployment(
    name="deployment name",
    model_uri=model_uri,
    # Config is optional
    config=dict(...),
)

# Predict
import pandas
df = pandas.DataFrame([
    {"a": 1,"b": 2,"c": 3},
    {"a": 4,"b": 5,"c": 6}
])
predictions = client.predict("deployment name", df)
```

## Model Registry plugin usage

Set the MLflow Model Registry URI to a directory in some Google Cloud Storage bucket, then log models using `mlflow.log_model` as usual.

```python
mlflow.set_registry_uri("gs://<bucket>/models/")
```
