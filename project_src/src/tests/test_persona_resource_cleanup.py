import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.persona_resource_cleanup import close_persona_resources, close_shared_memory_clients


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def shutdown(self, wait=False, cancel_futures=False):
        self.calls.append((wait, cancel_futures))


class FakeClient:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakeVectorStore:
    def __init__(self, client):
        self.client = client


class FakeMem0:
    def __init__(self, vector_client, telemetry_client=None):
        self.vector_store = FakeVectorStore(vector_client)
        self._telemetry_vector_store = FakeVectorStore(telemetry_client or vector_client)


class FakeMemory:
    def __init__(self, mem0=None):
        self.mem0 = mem0


class FakePersona:
    def __init__(self, mem0=None):
        self.close_calls = 0
        self._bg_executor = FakeExecutor()
        self.memory = FakeMemory(mem0)

    def close(self):
        self.close_calls += 1


class PersonaResourceCleanupTests(unittest.TestCase):
    def test_close_persona_resources_closes_persona_and_executor(self):
        persona = FakePersona()
        close_persona_resources(persona, close_persona=True)
        self.assertEqual(persona.close_calls, 1)
        self.assertEqual(persona._bg_executor.calls, [(False, False)])

    def test_close_persona_resources_can_skip_persona_close(self):
        persona = FakePersona()
        close_persona_resources(persona, close_persona=False)
        self.assertEqual(persona.close_calls, 0)
        self.assertEqual(persona._bg_executor.calls, [(False, False)])

    def test_close_shared_memory_clients_closes_each_client_once(self):
        vector_client = FakeClient()
        telemetry_client = FakeClient()
        shared_mem0 = FakeMem0(vector_client, telemetry_client)

        personas = [FakePersona(shared_mem0), FakePersona(shared_mem0)]
        close_shared_memory_clients(personas)

        self.assertEqual(vector_client.close_calls, 1)
        self.assertEqual(telemetry_client.close_calls, 1)

    def test_close_shared_memory_clients_handles_empty_inputs(self):
        close_shared_memory_clients([])


if __name__ == "__main__":
    unittest.main()
