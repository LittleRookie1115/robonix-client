# SPDX-License-Identifier: MulanPSL-2.0

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Awaitable, Callable, Iterable

AssetDownloader = Callable[[str, Path], Awaitable[None]]


def _positive_env_bytes(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_env_count(name: str, default: int) -> int:
    return _positive_env_bytes(name, default)


def default_urdf_cache_dir() -> Path:
    configured = os.environ.get("ROBONIX_CLIENT_MODEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    return cache_home / "robonix-client" / "urdf-assets"


@dataclass(frozen=True)
class CachedUrdfAsset:
    path: Path
    media_type: str
    size_bytes: int
    sha256: str

    @property
    def data(self) -> bytes:
        return self.path.read_bytes()


@dataclass(frozen=True)
class UrdfAssetMetadata:
    path: str
    size_bytes: int
    sha256: str
    media_type: str


@dataclass
class UrdfResourceSet:
    assets: dict[str, UrdfAssetMetadata]
    downloader: AssetDownloader | None


class AssetLimitError(ValueError):
    pass


class AssetDownloadError(RuntimeError):
    pass


class UrdfAssetStore:
    """Content-addressed disk cache for inline and lazily streamed URDF assets."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        max_file_bytes: int | None = None,
        max_model_bytes: int | None = None,
        max_cache_bytes: int | None = None,
        max_concurrent_downloads: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_urdf_cache_dir()
        self.max_file_bytes = self._configured_limit(
            max_file_bytes, "ROBONIX_CLIENT_MODEL_FILE_MAX_BYTES", 1024 * 1024 * 1024
        )
        self.max_model_bytes = self._configured_limit(
            max_model_bytes,
            "ROBONIX_CLIENT_MODEL_TOTAL_MAX_BYTES",
            4 * 1024 * 1024 * 1024,
        )
        self.max_cache_bytes = self._configured_limit(
            max_cache_bytes,
            "ROBONIX_CLIENT_MODEL_CACHE_MAX_BYTES",
            8 * 1024 * 1024 * 1024,
        )
        self.max_concurrent_downloads = self._configured_count(
            max_concurrent_downloads,
            "ROBONIX_CLIENT_MODEL_DOWNLOAD_CONCURRENCY",
            4,
        )
        self._resource_sets: dict[str, UrdfResourceSet] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._verified_files: dict[tuple[str, str], tuple[int, int, int]] = {}
        self._download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        self._lock = RLock()

    def put(self, assets: Iterable[tuple[str, bytes]]) -> str:
        """Persist one legacy inline response and return its content id."""
        normalized: dict[str, bytes] = {}
        for raw_path, raw_data in assets:
            path = self.normalize_path(raw_path)
            data = bytes(raw_data)
            existing = normalized.get(path)
            if existing is not None and existing != data:
                raise ValueError(f"conflicting URDF asset path: {path}")
            normalized[path] = data
        if not normalized:
            return ""

        metadata: list[UrdfAssetMetadata] = []
        for path, data in sorted(normalized.items()):
            metadata.append(
                UrdfAssetMetadata(
                    path=path,
                    size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    media_type=self.media_type(path),
                )
            )
        resource_set_id = self._manifest_digest(metadata)
        self.register(resource_set_id, metadata)
        for path, data in normalized.items():
            target = self._asset_path(resource_set_id, path)
            if not target.exists():
                self._atomic_write(target, data)
        self._enforce_cache_budget(protected={resource_set_id})
        return resource_set_id

    def register(
        self,
        resource_set_id: str,
        assets: Iterable[UrdfAssetMetadata],
        downloader: AssetDownloader | None = None,
    ) -> str:
        """Register a validated remote manifest without downloading its files."""
        resource_set_id = self.normalize_resource_set_id(resource_set_id)
        normalized: dict[str, UrdfAssetMetadata] = {}
        total_size = 0
        for asset in assets:
            path = self.normalize_path(asset.path)
            size_bytes = int(asset.size_bytes)
            if size_bytes < 0 or size_bytes > self.max_file_bytes:
                raise AssetLimitError(
                    f"URDF asset '{path}' size {size_bytes} exceeds limit {self.max_file_bytes}"
                )
            sha256 = str(asset.sha256).lower()
            if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
                raise ValueError(f"invalid SHA-256 for URDF asset '{path}'")
            item = UrdfAssetMetadata(
                path=path,
                size_bytes=size_bytes,
                sha256=sha256,
                media_type=asset.media_type or self.media_type(path),
            )
            if path in normalized and normalized[path] != item:
                raise ValueError(f"conflicting URDF asset metadata: {path}")
            normalized[path] = item
            total_size += size_bytes
        if total_size > self.max_model_bytes:
            raise AssetLimitError(
                f"URDF model size {total_size} exceeds limit {self.max_model_bytes}"
            )
        if total_size > self.max_cache_bytes:
            raise AssetLimitError(
                f"URDF model size {total_size} exceeds cache limit {self.max_cache_bytes}"
            )
        if self._manifest_digest(normalized.values()) != resource_set_id:
            raise ValueError("URDF resource set id does not match its manifest")
        with self._lock:
            self._resource_sets[resource_set_id] = UrdfResourceSet(normalized, downloader)
        return resource_set_id

    async def get_or_fetch(self, resource_set_id: str, path: str) -> CachedUrdfAsset:
        """Return a verified cache file, downloading it once when necessary."""
        resource_set_id = self.normalize_resource_set_id(resource_set_id)
        normalized_path = self.normalize_path(path)
        with self._lock:
            resource_set = self._resource_sets[resource_set_id]
            metadata = resource_set.assets[normalized_path]
            downloader = resource_set.downloader
            asset_lock = self._locks.setdefault(
                (resource_set_id, normalized_path), asyncio.Lock()
            )
        target = self._asset_path(resource_set_id, normalized_path)
        cache_key = (resource_set_id, normalized_path)
        async with asset_lock:
            if not self._is_valid_file(target, metadata, cache_key):
                if downloader is None:
                    raise KeyError(normalized_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(
                    f".{target.name}.{os.getpid()}.{time.time_ns()}.part"
                )
                try:
                    async with self._download_semaphore:
                        try:
                            await downloader(normalized_path, temporary)
                        except Exception as exc:
                            raise AssetDownloadError(
                                f"failed to download URDF asset: {normalized_path}"
                            ) from exc
                    if not self._is_valid_file(temporary, metadata):
                        raise AssetDownloadError(
                            f"downloaded URDF asset failed validation: {normalized_path}"
                        )
                    os.replace(temporary, target)
                    target_stat = target.stat()
                    with self._lock:
                        self._verified_files[cache_key] = (
                            target_stat.st_size,
                            target_stat.st_mtime_ns,
                            target_stat.st_ctime_ns,
                        )
                finally:
                    temporary.unlink(missing_ok=True)
                self._enforce_cache_budget(protected={resource_set_id})
            os.utime(target, None)
            target_stat = target.stat()
            with self._lock:
                self._verified_files[cache_key] = (
                    target_stat.st_size,
                    target_stat.st_mtime_ns,
                    target_stat.st_ctime_ns,
                )
        return CachedUrdfAsset(
            path=target,
            media_type=metadata.media_type,
            size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
        )

    def get(self, resource_set_id: str, path: str) -> CachedUrdfAsset:
        """Return an already materialized asset without network access."""
        resource_set_id = self.normalize_resource_set_id(resource_set_id)
        normalized_path = self.normalize_path(path)
        with self._lock:
            metadata = self._resource_sets[resource_set_id].assets[normalized_path]
        target = self._asset_path(resource_set_id, normalized_path)
        if not self._is_valid_file(
            target, metadata, (resource_set_id, normalized_path)
        ):
            raise KeyError(normalized_path)
        return CachedUrdfAsset(
            path=target,
            media_type=metadata.media_type,
            size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
        )

    def clear(self, *, remove_files: bool = False) -> None:
        """Drop registrations and optionally delete disk cache contents."""
        with self._lock:
            self._resource_sets.clear()
            self._locks.clear()
            self._verified_files.clear()
        if remove_files:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _asset_path(self, resource_set_id: str, path: str) -> Path:
        return self.cache_dir / resource_set_id / Path(*PurePosixPath(path).parts)

    @staticmethod
    def _configured_limit(value: int | None, name: str, default: int) -> int:
        configured = _positive_env_bytes(name, default) if value is None else value
        if configured < 1:
            raise ValueError(f"{name} must be positive")
        return configured

    @staticmethod
    def _configured_count(value: int | None, name: str, default: int) -> int:
        configured = _positive_env_count(name, default) if value is None else value
        if configured < 1:
            raise ValueError(f"{name} must be positive")
        return configured

    @staticmethod
    def _manifest_digest(assets: Iterable[UrdfAssetMetadata]) -> str:
        digest = hashlib.sha256()
        for asset in sorted(assets, key=lambda item: item.path):
            path_bytes = asset.path.encode("utf-8")
            digest.update(len(path_bytes).to_bytes(8, "big"))
            digest.update(path_bytes)
            digest.update(asset.size_bytes.to_bytes(8, "big"))
            digest.update(asset.sha256.encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _is_valid_file(
        self,
        path: Path,
        metadata: UrdfAssetMetadata,
        cache_key: tuple[str, str] | None = None,
    ) -> bool:
        if not path.is_file():
            return False
        file_stat = path.stat()
        fingerprint = (
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
        if file_stat.st_size != metadata.size_bytes:
            return False
        if cache_key is not None:
            with self._lock:
                if self._verified_files.get(cache_key) == fingerprint:
                    return True
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        valid = digest.hexdigest() == metadata.sha256
        if valid and cache_key is not None:
            with self._lock:
                self._verified_files[cache_key] = fingerprint
        return valid

    def _enforce_cache_budget(self, protected: set[str]) -> None:
        if not self.cache_dir.exists():
            return
        files = [
            path
            for path in self.cache_dir.rglob("*")
            if path.is_file() and not path.name.endswith(".part")
        ]
        total = sum(path.stat().st_size for path in files)
        if total <= self.max_cache_bytes:
            return
        files.sort(key=lambda path: path.stat().st_mtime)
        for path in files:
            relative = path.relative_to(self.cache_dir)
            resource_set_id = relative.parts[0] if relative.parts else ""
            if resource_set_id in protected:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            with self._lock:
                self._verified_files.pop(
                    (resource_set_id, relative.relative_to(resource_set_id).as_posix()),
                    None,
                )
            total -= size
            if total <= self.max_cache_bytes:
                break

    @staticmethod
    def media_type(path: str) -> str:
        if path.lower().endswith(".stl"):
            return "model/stl"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    @staticmethod
    def normalize_resource_set_id(raw_value: str) -> str:
        value = str(raw_value or "").strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("invalid URDF resource set id")
        return value

    @staticmethod
    def normalize_path(raw_path: str) -> str:
        """Accept only non-empty POSIX paths below the resource-set root."""
        value = str(raw_path or "").strip()
        path = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or any(part in {"", "."} for part in path.parts)
        ):
            raise ValueError(f"invalid URDF asset path: {raw_path!r}")
        return path.as_posix()


urdf_asset_store = UrdfAssetStore()
