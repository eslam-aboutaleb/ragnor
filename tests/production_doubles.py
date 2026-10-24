"""Nonfunctional ports for startup validation only; never production evidence."""


class VersionPort:
    async def find_active(self, session):
        raise NotImplementedError

    async def close_active(self, session):
        raise NotImplementedError

    async def create_version(self, session, snapshot):
        raise NotImplementedError

    async def delete_chunks(self, session, version_id):
        raise NotImplementedError


class StoragePort:
    async def upsert(self, documents, **kwargs):
        raise NotImplementedError

    async def count(self, **kwargs):
        raise NotImplementedError
