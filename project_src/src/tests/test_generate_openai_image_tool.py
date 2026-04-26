import base64
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.tools import generate_openai_image


class _FakeImages:
    def __init__(self):
        self.generate_calls = []
        self.edit_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"generated").decode("ascii"))])

    def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        for image_file in kwargs["image"]:
            image_file.read(1)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"edited").decode("ascii"))])


class _FakeOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.images = _FakeImages()
        self.__class__.instances.append(self)


def _fake_app_config():
    return SimpleNamespace(
        image_generation=SimpleNamespace(
            api_key=None,
            model="gpt-image-2",
            base_url="https://api.openai.com/v1",
            size="1024x1024",
            quality="auto",
            moderation="auto",
            background="auto",
            output_format="png",
            response_format="b64_json",
            n=1,
            output_compression=None,
            timeout=120,
            user=None,
            output_dir="./data/generated_images",
            output_prefix="gpt_image",
            max_retries=0,
            default_prompt="",
        )
    )


class GenerateOpenAIImageToolTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.instances = []

    def _run_main(self, args):
        with patch.object(sys, "argv", ["generate_openai_image.py", *args]), \
             patch.object(generate_openai_image, "OpenAI", _FakeOpenAI), \
             patch.object(generate_openai_image.AppConfig, "load", return_value=_fake_app_config()):
            return generate_openai_image.main()

    def test_prompt_only_uses_generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "generated.png"

            result = self._run_main([
                "make a small orange pixel sprite",
                "--api-key", "test-key",
                "--output", str(output),
            ])

            client = _FakeOpenAI.instances[0]
            self.assertEqual(result, 0)
            self.assertEqual(output.read_bytes(), b"generated")
            self.assertEqual(len(client.images.generate_calls), 1)
            self.assertEqual(len(client.images.edit_calls), 0)
            self.assertEqual(client.images.generate_calls[0]["moderation"], "auto")

    def test_reference_images_use_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_a = root / "ref-a.png"
            ref_b = root / "ref-b.jpg"
            output = root / "edited.png"
            ref_a.write_bytes(b"a")
            ref_b.write_bytes(b"b")

            result = self._run_main([
                "use these references to generate a new sprite",
                "--api-key", "test-key",
                "--input-image", str(ref_a),
                "--input-image", str(ref_b),
                "--input-fidelity", "high",
                "--output", str(output),
            ])

            client = _FakeOpenAI.instances[0]
            self.assertEqual(result, 0)
            self.assertEqual(output.read_bytes(), b"edited")
            self.assertEqual(len(client.images.generate_calls), 0)
            self.assertEqual(len(client.images.edit_calls), 1)
            call = client.images.edit_calls[0]
            self.assertEqual(call["prompt"], "use these references to generate a new sprite")
            self.assertEqual(call["input_fidelity"], "high")
            self.assertEqual(len(call["image"]), 2)
            self.assertNotIn("moderation", call)

    def test_missing_reference_image_fails_before_api_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.png"
            output = Path(tmpdir) / "edited.png"

            with self.assertRaises(FileNotFoundError):
                self._run_main([
                    "use this reference",
                    "--api-key", "test-key",
                    "--input-image", str(missing),
                    "--output", str(output),
                ])

            self.assertEqual(len(_FakeOpenAI.instances), 1)
            client = _FakeOpenAI.instances[0]
            self.assertEqual(len(client.images.generate_calls), 0)
            self.assertEqual(len(client.images.edit_calls), 0)


if __name__ == "__main__":
    unittest.main()
