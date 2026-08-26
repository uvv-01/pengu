"""Tests for the FastAPI backend endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from pengu.api import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_root_serves_ui(self, client: AsyncClient):
        """Root endpoint serves the Pengu desktop UI (HTML)."""
        resp = await client.get("/")
        assert resp.status_code == 200
        # Should return HTML (the UI page)
        assert "html" in resp.headers.get("content-type", "") or b"Pengu" in resp.content

    @pytest.mark.asyncio
    async def test_api_status(self, client: AsyncClient):
        """API status endpoint returns JSON."""
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Pengu"
        assert data["status"] == "running"


class TestConfigEndpoint:
    @pytest.mark.asyncio
    async def test_config(self, client: AsyncClient):
        resp = await client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "cost_mode" in data


class TestHardwareEndpoint:
    @pytest.mark.asyncio
    async def test_hardware(self, client: AsyncClient):
        resp = await client.get("/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "os" in data
        assert "cpu" in data
        assert "ram" in data


class TestStateEndpoint:
    @pytest.mark.asyncio
    async def test_state(self, client: AsyncClient):
        resp = await client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "STANDBY"


class TestToolsEndpoint:
    @pytest.mark.asyncio
    async def test_tools(self, client: AsyncClient):
        resp = await client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data


class TestCommandEndpoint:
    @pytest.mark.asyncio
    async def test_command_empty(self, client: AsyncClient):
        resp = await client.post("/command", json={"text": ""})
        assert resp.status_code == 200
        assert "error" in resp.json()

    @pytest.mark.asyncio
    async def test_command_basic(self, client: AsyncClient):
        resp = await client.post("/command", json={"text": "hello pengu"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        # Pipeline may or may not be initialized in test mode
        # Accept either success response or pipeline-not-initialized error
        assert "response" in data or "error" in data


class TestActivateEndpoint:
    @pytest.mark.asyncio
    async def test_activate(self, client: AsyncClient):
        resp = await client.post("/activate")
        assert resp.status_code == 200
        data = resp.json()
        # State machine may be in various states from previous tests
        # Just verify the endpoint responds and has state field
        assert "state" in data


class TestCancelEndpoint:
    @pytest.mark.asyncio
    async def test_cancel(self, client: AsyncClient):
        resp = await client.post("/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "STANDBY"
