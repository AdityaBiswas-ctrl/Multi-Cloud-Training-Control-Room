# Multi-Cloud Prefect Training Pipeline

This project scaffolds an end-to-end ML training pipeline with:

- Prefect orchestration
- Cost-aware routing across AWS, GCP, and Azure
- Automatic failover between cloud providers
- Local simulation mode for development without cloud credentials
- Clear extension points for real cloud training backends

## What the pipeline does

For each training job, the flow:

1. Scores each cloud by estimated run cost, health, capacity, and region fit
2. Selects the cheapest healthy provider that stays inside budget
3. Submits the training job
4. Fails over to the next ranked provider if the attempt fails for a recoverable reason
5. Evaluates the resulting model and emits a machine-readable summary

## Project layout

```text
src/multicloud_ml_pipeline/
  cli.py
  config.py
  exceptions.py
  flow.py
  models.py
  prefect_compat.py
  router.py
  providers/
    aws.py
    azure.py
    base.py
    factory.py
    gcp.py
examples/
  pipeline-config.json
tests/
  test_failover.py
  test_router.py
```

## Quickstart

Install dependencies:

```powershell
py -3.13 -m pip install -e .
```

Run the sample pipeline:

```powershell
py -3.13 -m multicloud_ml_pipeline.cli --config examples/pipeline-config.json
```

Run the browser UI locally:

```powershell
$env:PYTHONPATH="src"
py -3.13 -m multicloud_ml_pipeline.web --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser and click `Run Pipeline`.

Run tests:

```powershell
$env:PYTHONPATH="src"
py -3.13 -m unittest discover -s tests -v
```

## Simulation vs real cloud execution

The sample config uses `simulation` mode so the flow can run locally.

Each provider adapter also exposes a `build_submission_payload` method for the real backend it targets:

- AWS: SageMaker-style training payload
- GCP: Vertex AI custom job payload
- Azure: Azure ML command job payload

To connect this scaffold to production infrastructure:

1. Switch the provider `mode` from `simulation` to `sdk`
2. Install the relevant optional dependencies
3. Add credentials for the target cloud
4. Replace the placeholder `submit_sdk_job` implementation in each provider adapter with your environment-specific submission call

## Sample behavior

The example configuration intentionally forces AWS to fail its first attempt so the flow reroutes to GCP. Azure stays available as the third fallback target.

## Design notes

- Recoverable failures such as capacity exhaustion, transient control-plane errors, and timeouts trigger failover.
- Fatal failures such as invalid training specs stop the flow immediately because retrying on another cloud would not help.
- Budget limits are enforced before submission. Providers whose estimated cost exceeds the job budget are excluded from routing.
