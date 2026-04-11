import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.grpc_contract import (
    PREDICT_BATCH_METHOD,
    PREDICT_METHOD,
    predict_batch_request_class,
    predict_batch_response_class,
    predict_request_class,
    predict_response_class,
)


class GrpcContractTests(unittest.TestCase):
    def test_method_paths(self):
        self.assertEqual(PREDICT_METHOD, "/bert_inference.BERTInference/Predict")
        self.assertEqual(PREDICT_BATCH_METHOD, "/bert_inference.BERTInference/PredictBatch")

    def test_predict_message_classes_have_expected_fields(self):
        request_cls = predict_request_class()
        response_cls = predict_response_class()

        self.assertEqual(
            set(request_cls.DESCRIPTOR.fields_by_name),
            {"input_ids", "attention_mask", "personality"},
        )
        self.assertEqual(
            set(response_cls.DESCRIPTOR.fields_by_name),
            {
                "emotion_logits",
                "behavior_logits",
                "tone_logits",
                "intensity",
                "response_length_logits",
                "error",
            },
        )

    def test_predict_batch_roundtrip(self):
        request_cls = predict_batch_request_class()
        response_cls = predict_batch_response_class()

        request = request_cls()
        request.input_ids.extend([101, 102, 201, 202])
        request.attention_mask.extend([1, 1, 1, 1])
        request.personality.extend([0.1, 0.2, 0.3])
        request.batch_size = 2
        request.seq_length = 2

        encoded_request = request.SerializeToString()
        decoded_request = request_cls.FromString(encoded_request)
        self.assertEqual(list(decoded_request.input_ids), [101, 102, 201, 202])
        self.assertEqual(list(decoded_request.attention_mask), [1, 1, 1, 1])
        self.assertEqual(len(decoded_request.personality), 3)
        for actual, expected in zip(decoded_request.personality, [0.1, 0.2, 0.3]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(decoded_request.batch_size, 2)
        self.assertEqual(decoded_request.seq_length, 2)

        response = response_cls()
        response.emotion_logits.extend([0.1, 0.2])
        response.behavior_logits.extend([0.3, 0.4])
        response.tone_logits.extend([0.5, 0.6])
        response.intensity.extend([0.7])
        response.response_length_logits.extend([0.8, 0.9, 1.0])
        response.error = ""

        encoded_response = response.SerializeToString()
        decoded_response = response_cls.FromString(encoded_response)
        self.assertEqual(len(decoded_response.response_length_logits), 3)
        for actual, expected in zip(decoded_response.response_length_logits, [0.8, 0.9, 1.0]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(decoded_response.error, "")


if __name__ == "__main__":
    unittest.main()
