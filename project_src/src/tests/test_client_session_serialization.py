import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from src.client.session import ClientChatSession
from src.client.vision_runtime import VisionRuntimeController
from src.core_engine.api import DirectRuntime
from src.vision import VisualEvent


class _ConcurrentTarget:
    def __init__(self, *, block_first: bool = False):
        self.personality = SimpleNamespace(name="concurrent-target")
        self.llm_client = SimpleNamespace(model="fake-model")
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.calls = []
        self.closed = False
        self.close_time = None
        self.chat_end_time = None
        self.block_first = block_first
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def chat(self, text, **kwargs):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append((text, kwargs))
            is_first = len(self.calls) == 1

        if self.block_first and is_first:
            self.first_started.set()
            self.release_first.wait(timeout=2.0)
        else:
            time.sleep(0.05)

        with self._lock:
            self._active -= 1
            self.chat_end_time = time.monotonic()

        return {
            "response": f"ok:{text}",
            "should_respond": True,
            "emotion": {},
            "behavior": {},
        }

    def close(self):
        with self._lock:
            self.closed = True
            self.close_time = time.monotonic()


class ClientChatSessionSerializationTests(unittest.TestCase):
    def test_same_session_send_message_is_serialized(self):
        target = _ConcurrentTarget()
        session = ClientChatSession.from_target(DirectRuntime(target), context_id="ctx")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(session.send_message, "first"),
                executor.submit(session.send_message, "second"),
            ]
            replies = [future.result(timeout=2.0).reply for future in futures]

        self.assertEqual(target.max_active, 1)
        self.assertCountEqual(replies, ["ok:first", "ok:second"])
        self.assertEqual(len(target.calls), 2)

    def test_different_sessions_can_run_concurrently(self):
        target = _ConcurrentTarget()
        runtime = DirectRuntime(target)
        first = ClientChatSession.from_target(runtime, context_id="ctx-a")
        second = ClientChatSession.from_target(runtime, context_id="ctx-b")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(first.send_message, "first"),
                executor.submit(second.send_message, "second"),
            ]
            for future in futures:
                future.result(timeout=2.0)

        self.assertGreaterEqual(target.max_active, 2)
        self.assertEqual(len(target.calls), 2)

    def test_shutdown_waits_for_active_send_before_closing_runtime(self):
        target = _ConcurrentTarget(block_first=True)
        session = ClientChatSession.from_target(DirectRuntime(target), context_id="ctx")

        send_thread = threading.Thread(target=session.send_message, args=("blocked",))
        send_thread.start()
        self.assertTrue(target.first_started.wait(timeout=1.0))

        shutdown_returned = threading.Event()

        def _shutdown():
            session.shutdown()
            shutdown_returned.set()

        shutdown_thread = threading.Thread(target=_shutdown)
        shutdown_thread.start()
        time.sleep(0.05)

        self.assertFalse(shutdown_returned.is_set())
        self.assertFalse(target.closed)

        target.release_first.set()
        send_thread.join(timeout=1.0)
        shutdown_thread.join(timeout=1.0)

        self.assertTrue(shutdown_returned.is_set())
        self.assertTrue(target.closed)
        self.assertIsNotNone(target.chat_end_time)
        self.assertIsNotNone(target.close_time)
        self.assertGreaterEqual(target.close_time, target.chat_end_time)

    def test_shutdown_rejects_new_send_message(self):
        target = _ConcurrentTarget()
        session = ClientChatSession.from_target(DirectRuntime(target), context_id="ctx")

        session.shutdown()

        with self.assertRaises(RuntimeError):
            session.send_message("after close")

    def test_shutdown_waits_for_queued_send(self):
        target = _ConcurrentTarget(block_first=True)
        session = ClientChatSession.from_target(DirectRuntime(target), context_id="ctx")

        first_thread = threading.Thread(target=session.send_message, args=("first",))
        first_thread.start()
        self.assertTrue(target.first_started.wait(timeout=1.0))

        second_result = []
        second_thread = threading.Thread(
            target=lambda: second_result.append(session.send_message("second").reply)
        )
        second_thread.start()
        time.sleep(0.05)

        shutdown_returned = threading.Event()
        shutdown_thread = threading.Thread(
            target=lambda: (session.shutdown(), shutdown_returned.set())
        )
        shutdown_thread.start()
        time.sleep(0.05)

        self.assertFalse(shutdown_returned.is_set())
        self.assertFalse(target.closed)

        target.release_first.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)
        shutdown_thread.join(timeout=1.0)

        self.assertEqual(second_result, ["ok:second"])
        self.assertTrue(target.closed)
        self.assertTrue(shutdown_returned.is_set())
        self.assertEqual([text for text, _ in target.calls], ["first", "second"])


class _BlockingSession:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.send_count = 0
        self._lock = threading.Lock()

    def send_message(self, text, **kwargs):
        with self._lock:
            self.send_count += 1
        self.started.set()
        self.release.wait(timeout=2.0)
        return SimpleNamespace(reply="visual reply")


class VisionRuntimeShutdownTests(unittest.TestCase):
    def test_shutdown_waits_for_active_visual_direct_reply(self):
        target = _ConcurrentTarget()
        runtime = DirectRuntime(target)
        session = _BlockingSession()
        controller = VisionRuntimeController.from_runtime(runtime, session)  # type: ignore[arg-type]
        event = VisualEvent(
            event_id="visual-1",
            peak_frame_index=1,
            timestamp=0.1,
            peak_score=1.0,
            representative_frame_index=1,
        )

        controller._handle_promoted_event(event)
        self.assertTrue(session.started.wait(timeout=1.0))

        shutdown_returned = threading.Event()
        shutdown_thread = threading.Thread(target=lambda: (controller.shutdown(), shutdown_returned.set()))
        shutdown_thread.start()
        time.sleep(0.05)

        self.assertFalse(shutdown_returned.is_set())
        session.release.set()
        shutdown_thread.join(timeout=1.0)

        self.assertTrue(shutdown_returned.is_set())
        self.assertEqual(session.send_count, 1)

    def test_promoted_event_after_shutdown_is_ignored(self):
        target = _ConcurrentTarget()
        runtime = DirectRuntime(target)
        session = _BlockingSession()
        controller = VisionRuntimeController.from_runtime(runtime, session)  # type: ignore[arg-type]
        event = VisualEvent(
            event_id="visual-1",
            peak_frame_index=1,
            timestamp=0.1,
            peak_score=1.0,
            representative_frame_index=1,
        )

        controller.shutdown()
        controller._handle_promoted_event(event)
        time.sleep(0.05)

        self.assertEqual(session.send_count, 0)


if __name__ == "__main__":
    unittest.main()
