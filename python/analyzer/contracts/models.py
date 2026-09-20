"""Local model inventory. Artifact verification is separate from runtime inference."""

from typing import Literal

from pydantic import BaseModel, Field


class ManagedModel(BaseModel):
    name: str
    version: str = Field(description="Artifact identity: sha256 of the manifest-pinned weights.")
    backend: str
    device: str = Field(description="Device selected by this backend's runtime policy.")
    input_requirements: list[str]
    size_bytes: int
    required: bool
    state: Literal["missing", "verified", "mismatch"]
    installed_sha256: str | None = None
    detail: str


class ModelInventory(BaseModel):
    default_pose_model: str
    models: list[ManagedModel]
