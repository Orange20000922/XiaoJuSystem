from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
PROTO_DIR = AGENT_ROOT / "proto"
LOCAL_PROTO = PROTO_DIR / "bert_inference.proto"
LOCAL_DESCRIPTOR = PROTO_DIR / "bert_inference.desc"
DEFAULT_BACKEND_PROTO = REPO_ROOT.parent / "AgentBackendPredict" / "proto" / "bert_inference.proto"
DEFAULT_BACKEND_PROTOC = (
    REPO_ROOT.parent
    / "AgentBackendPredict"
    / "vcpkg_installed"
    / "x64-windows"
    / "tools"
    / "protobuf"
    / "protoc.exe"
)


def sync_proto(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Proto source not found: {source}")
    PROTO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, LOCAL_PROTO)
    print(f"Synced proto from {source}")


def descriptor_signature(path: Path):
    file_set = descriptor_pb2.FileDescriptorSet()
    file_set.ParseFromString(path.read_bytes())

    def field_sig(field):
        return {
            "name": field.name,
            "number": field.number,
            "label": field.label,
            "type": field.type,
            "type_name": field.type_name,
        }

    def message_sig(message):
        return {
            "name": message.name,
            "fields": [field_sig(field) for field in message.field],
            "nested_types": [message_sig(nested) for nested in message.nested_type],
        }

    def service_sig(service):
        return {
            "name": service.name,
            "methods": [
                {
                    "name": method.name,
                    "input_type": method.input_type,
                    "output_type": method.output_type,
                    "client_streaming": method.client_streaming,
                    "server_streaming": method.server_streaming,
                }
                for method in service.method
            ],
        }

    return [
        {
            "name": file_proto.name,
            "package": file_proto.package,
            "syntax": file_proto.syntax,
            "messages": [message_sig(message) for message in file_proto.message_type],
            "services": [service_sig(service) for service in file_proto.service],
        }
        for file_proto in sorted(file_set.file, key=lambda item: item.name)
    ]


def find_fallback_protoc() -> Path | None:
    if DEFAULT_BACKEND_PROTOC.exists():
        return DEFAULT_BACKEND_PROTOC

    protoc = shutil.which("protoc")
    if protoc:
        return Path(protoc)
    return None


def compile_descriptor(output_path: Path, protoc_override: str | None = None) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from grpc_tools import protoc as grpc_protoc
    except ImportError:
        grpc_protoc = None

    if grpc_protoc is not None:
        args = [
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--descriptor_set_out={output_path}",
            "--include_imports",
            str(LOCAL_PROTO),
        ]
        exit_code = grpc_protoc.main(args)
        if exit_code != 0:
            raise RuntimeError(f"grpc_tools.protoc failed with exit code {exit_code}")
        return "grpc_tools.protoc"

    protoc_path = Path(protoc_override) if protoc_override else find_fallback_protoc()
    if protoc_path is None:
        raise RuntimeError(
            "No protoc compiler available. Install grpcio-tools, install protoc, or pass --protoc."
        )

    subprocess.run(
        [
            str(protoc_path),
            f"-I{PROTO_DIR}",
            f"--descriptor_set_out={output_path}",
            "--include_imports",
            str(LOCAL_PROTO),
        ],
        check=True,
    )
    return str(protoc_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate the Python-side gRPC contract descriptor.")
    parser.add_argument("--sync-from", help="Copy bert_inference.proto from an explicit source path before compiling.")
    parser.add_argument(
        "--sync-from-backend",
        action="store_true",
        help="Copy bert_inference.proto from the sibling AgentBackendPredict repo before compiling.",
    )
    parser.add_argument("--check", action="store_true", help="Validate that the checked-in descriptor matches the current proto semantics.")
    parser.add_argument("--protoc", help="Use an explicit protoc executable when grpcio-tools is unavailable.")
    args = parser.parse_args()

    if args.sync_from and args.sync_from_backend:
        parser.error("Use either --sync-from or --sync-from-backend, not both.")

    if args.sync_from:
        sync_proto(Path(args.sync_from).resolve())
    elif args.sync_from_backend:
        sync_proto(DEFAULT_BACKEND_PROTO)

    if not LOCAL_PROTO.exists():
        raise FileNotFoundError(f"Proto file not found: {LOCAL_PROTO}")

    if args.check:
        if not LOCAL_DESCRIPTOR.exists():
            raise FileNotFoundError(f"Descriptor file not found: {LOCAL_DESCRIPTOR}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_descriptor = Path(temp_dir) / LOCAL_DESCRIPTOR.name
            tool = compile_descriptor(temp_descriptor, args.protoc)
            if descriptor_signature(LOCAL_DESCRIPTOR) != descriptor_signature(temp_descriptor):
                print(
                    "Checked-in descriptor is stale. Run `python project_src/scripts/generate_grpc_contract.py` and commit the result.",
                    file=sys.stderr,
                )
                return 1
            print(f"Descriptor check passed via {tool}")
            return 0

    tool = compile_descriptor(LOCAL_DESCRIPTOR, args.protoc)
    print(f"Wrote descriptor to {LOCAL_DESCRIPTOR} via {tool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

