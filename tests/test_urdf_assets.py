# SPDX-License-Identifier: MulanPSL-2.0

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from robonix_client.urdf_assets import (
    AssetDownloadError,
    AssetLimitError,
    UrdfAssetMetadata,
    UrdfAssetStore,
)
from robonix_client.vitals_api import urdf_asset


class UrdfAssetStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = UrdfAssetStore(cache_dir=self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_caches_relative_resources_by_content(self) -> None:
        first_id = self.store.put([("meshes/base.stl", b"solid base")])
        second_id = self.store.put([("meshes/base.stl", b"solid base")])
        asset = self.store.get(first_id, "meshes/base.stl")

        self.assertEqual(first_id, second_id)
        self.assertEqual(asset.data, b"solid base")
        self.assertEqual(asset.media_type, "model/stl")

    def test_rejects_paths_outside_resource_root(self) -> None:
        with self.assertRaises(ValueError):
            self.store.put([("../secret.stl", b"secret")])
        with self.assertRaises(ValueError):
            self.store.put([("/tmp/secret.stl", b"secret")])
        with self.assertRaises(ValueError):
            self.store.put([(".", b"root")])

    def test_evicts_old_files_when_disk_budget_is_exceeded(self) -> None:
        store = UrdfAssetStore(cache_dir=self.directory.name, max_cache_bytes=3)
        old_id = store.put([("old.stl", b"old")])
        new_id = store.put([("new.stl", b"new")])

        with self.assertRaises(KeyError):
            store.get(old_id, "old.stl")
        self.assertEqual(store.get(new_id, "new.stl").data, b"new")

    def test_rejects_manifest_above_configured_model_limit(self) -> None:
        store = UrdfAssetStore(cache_dir=self.directory.name, max_model_bytes=4)
        metadata = UrdfAssetMetadata(
            path="large.stl",
            size_bytes=5,
            sha256=hashlib.sha256(b"large").hexdigest(),
            media_type="model/stl",
        )
        resource_set_id = store._manifest_digest([metadata])

        with self.assertRaises(AssetLimitError):
            store.register(resource_set_id, [metadata])

    def test_rejects_manifest_with_wrong_resource_set_id(self) -> None:
        metadata = UrdfAssetMetadata(
            path="base.stl",
            size_bytes=4,
            sha256=hashlib.sha256(b"base").hexdigest(),
            media_type="model/stl",
        )

        with self.assertRaises(ValueError):
            self.store.register("0" * 64, [metadata])


class UrdfAssetLazyDownloadTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        self.directory.cleanup()

    async def test_downloads_once_and_verifies_content(self) -> None:
        store = UrdfAssetStore(cache_dir=self.directory.name)
        data = b"solid streamed\n"
        metadata = UrdfAssetMetadata(
            path="meshes/base.stl",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            media_type="model/stl",
        )
        calls = 0

        async def download(_path: str, destination: Path) -> None:
            nonlocal calls
            calls += 1
            destination.write_bytes(data)

        resource_set_id = store._manifest_digest([metadata])
        store.register(resource_set_id, [metadata], download)
        first, second = await asyncio.gather(
            store.get_or_fetch(resource_set_id, metadata.path),
            store.get_or_fetch(resource_set_id, metadata.path),
        )

        self.assertEqual(calls, 1)
        self.assertEqual(first.data, data)
        self.assertEqual(second.path, first.path)

    async def test_rejects_download_with_wrong_hash(self) -> None:
        store = UrdfAssetStore(cache_dir=self.directory.name)
        metadata = UrdfAssetMetadata(
            path="meshes/base.stl",
            size_bytes=4,
            sha256=hashlib.sha256(b"good").hexdigest(),
            media_type="model/stl",
        )

        async def download(_path: str, destination: Path) -> None:
            destination.write_bytes(b"evil")

        resource_set_id = store._manifest_digest([metadata])
        store.register(resource_set_id, [metadata], download)
        with self.assertRaises(AssetDownloadError):
            await store.get_or_fetch(resource_set_id, metadata.path)


class UrdfAssetRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = UrdfAssetStore(cache_dir=self.directory.name)
        self.store_patch = patch(
            "robonix_client.vitals_api.urdf_asset_store", self.store
        )
        self.store_patch.start()

    def tearDown(self) -> None:
        self.store_patch.stop()
        self.directory.cleanup()

    def test_serves_cached_resource_with_immutable_headers(self) -> None:
        resource_set_id = self.store.put([("meshes/base.stl", b"solid base")])

        response = asyncio.run(urdf_asset(resource_set_id, "meshes/base.stl"))

        self.assertEqual(Path(response.path).read_bytes(), b"solid base")
        self.assertIn("immutable", response.headers["cache-control"])
        self.assertIn("etag", response.headers)

    def test_missing_resource_returns_not_found(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(urdf_asset("missing", "meshes/base.stl"))

        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
