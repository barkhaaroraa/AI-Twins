"""Blackboard registry: lazy per-(user, agent) instantiation + 1 Hz decay tick."""
import logging
import threading
import time
from typing import Dict, Optional, Tuple

from app.cognition.blackboard import Blackboard

log = logging.getLogger(__name__)

_registry: Dict[Tuple[str, str], Blackboard] = {}
_registry_lock = threading.RLock()

_decay_thread: Optional[threading.Thread] = None
_decay_stop = threading.Event()


def get_blackboard(user_id: str, agent_name: str) -> Blackboard:
    key = (user_id, agent_name)
    with _registry_lock:
        bb = _registry.get(key)
        if bb is None:
            bb = Blackboard(user_id=user_id, agent_name=agent_name)
            _registry[key] = bb
        return bb


def all_blackboards():
    with _registry_lock:
        return list(_registry.values())


def _decay_loop():
    last = time.monotonic()
    while not _decay_stop.is_set():
        now = time.monotonic()
        dt = now - last
        last = now
        try:
            for bb in all_blackboards():
                bb.decay(dt)
        except Exception:
            log.exception("decay tick error")
        _decay_stop.wait(1.0)


def start_decay_tick():
    global _decay_thread
    if _decay_thread is not None and _decay_thread.is_alive():
        return
    _decay_stop.clear()
    _decay_thread = threading.Thread(target=_decay_loop, daemon=True, name="bb-decay")
    _decay_thread.start()


def stop_decay_tick():
    _decay_stop.set()
