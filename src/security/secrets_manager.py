"""Secrets manager abstraction for production-safe config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretRef:
    name: str
    required: bool = True
    default: str | None = None


class SecretResolver:
    """Resolve secrets from env / AWS Secrets Manager / Vault.

    Priority:
      1. Environment variable
      2. AWS Secrets Manager (optional, if boto3 available)
      3. HashiCorp Vault (optional, if hvac available)
      4. default (if provided and not required)
    """

    def __init__(self) -> None:
        self._aws_enabled = bool(os.getenv("AWS_REGION"))
        self._vault_enabled = bool(os.getenv("VAULT_ADDR"))

    def get(self, ref: SecretRef) -> str:
        env_val = os.getenv(ref.name)
        if env_val:
            return env_val

        if self._aws_enabled:
            aws_val = self._from_aws(ref.name)
            if aws_val:
                return aws_val

        if self._vault_enabled:
            vault_val = self._from_vault(ref.name)
            if vault_val:
                return vault_val

        if ref.default is not None:
            return ref.default

        if ref.required:
            raise RuntimeError(f"Secret {ref.name} not found in env/AWS/Vault")
        return ""

    def _from_aws(self, name: str) -> str | None:
        try:
            import boto3

            region = os.getenv("AWS_REGION")
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=name)
            return resp.get("SecretString")
        except Exception:
            return None

    def _from_vault(self, name: str) -> str | None:
        try:
            import hvac

            addr = os.getenv("VAULT_ADDR")
            token = os.getenv("VAULT_TOKEN")
            mount = os.getenv("VAULT_MOUNT", "secret")
            client = hvac.Client(url=addr, token=token)
            data = client.secrets.kv.v2.read_secret_version(
                path=name,
                mount_point=mount,
            )
            return str(data["data"]["data"].get("value") or "")
        except Exception:
            return None
