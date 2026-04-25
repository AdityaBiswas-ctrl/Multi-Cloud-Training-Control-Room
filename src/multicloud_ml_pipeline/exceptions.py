class PipelineError(Exception):
    """Base pipeline exception."""


class RoutingError(PipelineError):
    """Raised when no cloud provider is eligible for a job."""


class TrainingJobError(PipelineError):
    """Base error for a provider submission failure."""


class RecoverableTrainingError(TrainingJobError):
    """A temporary failure that should trigger failover."""


class FatalTrainingError(TrainingJobError):
    """A non-recoverable failure that should stop the flow."""
