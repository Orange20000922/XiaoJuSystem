"""Persona lifecycle registry."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from src.core_engine.persona import PersonaInstance
from src.core_engine.persona_resource_cleanup import (
    close_persona_resources,
    close_shared_memory_clients,
)
from src.core_engine.shared_infra import SharedInfra
from configs.model_config import SchedulerConfig
from configs.config_loader import AppConfig


@dataclass
class RegisteredPersona:
    name: str
    persona: PersonaInstance
    config_source: str = ""


class PersonaRegistry:
    def __init__(
        self,
        infra: SharedInfra,
        config: Optional[SchedulerConfig] = None,
    ):
        self._infra = infra
        self._config = config or SchedulerConfig()
        self._personas: Dict[str, RegisteredPersona] = {}
        self._lock = threading.Lock()

        from threading import BoundedSemaphore

        self._llm_semaphore = BoundedSemaphore(self._config.max_concurrent_llm)
        infra.llm_semaphore = self._llm_semaphore
        infra.llm_acquire_timeout = self._config.llm_acquire_timeout

        logger.info(
            "PersonaRegistry initialized: "
            f"max_concurrent_llm={self._config.max_concurrent_llm} "
            f"llm_acquire_timeout={self._config.llm_acquire_timeout}"
        )

    def register(
        self,
        app_config: AppConfig,
        config_source: str = "",
    ) -> RegisteredPersona:
        name = app_config.personality.name

        with self._lock:
            if name in self._personas:
                raise ValueError(f"persona '{name}' is already registered")

        persona = PersonaInstance.from_app_config(self._infra, app_config)
        persona.muti_persona_mode = True
        registered = RegisteredPersona(name=name, persona=persona, config_source=config_source)

        with self._lock:
            if name in self._personas:
                close_persona_resources(persona, close_persona=True)
                raise ValueError(f"persona '{name}' was registered concurrently")
            self._personas[name] = registered

        logger.info(f"Registered persona '{name}' (source={config_source})")
        return registered

    def register_all(
        self,
        configs: List[Tuple[AppConfig, str]],
        max_workers: Optional[int] = None,
    ) -> List[RegisteredPersona]:
        if not configs:
            return []

        workers = max_workers or len(configs)

        def _register_one(args):
            app_config, config_source = args
            name = app_config.personality.name
            try:
                registered = self.register(app_config, config_source)
                return name, registered, None
            except Exception as exc:
                logger.error(
                    f"Failed to register persona '{name}' in parallel: {exc}",
                    exc_info=True,
                )
                return name, None, exc

        results: List[RegisteredPersona] = []
        failures: List[str] = []

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="persona-init") as pool:
            futures = {pool.submit(_register_one, cfg): cfg[0].personality.name for cfg in configs}
            for future in as_completed(futures):
                name, registered, _ = future.result()
                if registered is not None:
                    results.append(registered)
                else:
                    failures.append(name)

        if failures:
            logger.warning(f"register_all failures: {failures}")
        logger.info(f"register_all completed: {len(results)}/{len(configs)} personas registered")

        if results or not configs:
            return results
        raise ValueError(f"All persona registrations failed: {failures}")

    def unregister(self, name: str, *, close_persona: bool = True) -> bool:
        with self._lock:
            registered = self._personas.pop(name, None)
        if registered is None:
            logger.warning(f"unregister: persona '{name}' not found")
            return False

        logger.info(f"Unregistering persona '{name}'...")
        close_persona_resources(registered.persona, close_persona=close_persona)
        logger.info(f"Persona '{name}' unregistered")
        return True

    def get_persona(self, name: str) -> Optional[RegisteredPersona]:
        with self._lock:
            return self._personas.get(name)

    @property
    def persona_names(self) -> List[str]:
        with self._lock:
            return list(self._personas.keys())

    def get_health_report(self) -> Dict[str, Dict]:
        report = {}
        with self._lock:
            for name, registered in self._personas.items():
                report[name] = {
                    "alive": True,
                    "config_source": registered.config_source,
                }
        return report

    def shutdown(self, *, close_persona: bool = True):
        with self._lock:
            snapshot = list(self._personas.values())
            self._personas.clear()

        try:
            for registered in snapshot:
                logger.info(f"Shutting down persona '{registered.name}'...")
                close_persona_resources(registered.persona, close_persona=close_persona)
            close_shared_memory_clients(registered.persona for registered in snapshot)
        finally:
            self._infra.close()

        logger.info("PersonaRegistry fully shut down")

    def __len__(self) -> int:
        with self._lock:
            return len(self._personas)

    def __repr__(self) -> str:
        with self._lock:
            names = list(self._personas.keys())
        return f"PersonaRegistry(personas={names})"
