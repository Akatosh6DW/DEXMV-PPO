"""Small pipe-only process pool for independent native soft-body states."""

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import threading

from s12_physics_worker import read_message, write_message


class PhysicsPool:
    def __init__(self, cfg, count):
        if count<1 or count>8:
            raise ValueError("Use 1..8 isolated physics workers")
        self.processes = []
        self.lock = threading.RLock()
        self.threads = ThreadPoolExecutor(max_workers=count)
        atexit.register(self.close)
        try:
            for _ in range(count):
                p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("s12_physics_worker.py"))],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                self.processes.append(p)
                write_message(p.stdin, asdict(cfg))
                ready = read_message(p.stdout)
                if not ready["ready"] or ready["config"] != asdict(cfg):
                    raise RuntimeError("Physics worker handshake mismatch")
        except BaseException:
            self.close()
            raise

    def map(self, mode, batch):
        with self.lock:
            chunks = [batch[i::len(self.processes)] for i in range(len(self.processes))]

            def request(item):
                process, chunk = item
                if not chunk:
                    return []
                write_message(process.stdin, (mode, chunk))
                result = read_message(process.stdout)
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result["result"]

            chunks = list(self.threads.map(request, zip(self.processes, chunks)))
            return [chunks[i%len(chunks)][i//len(chunks)] for i in range(len(batch))]

    def close(self):
        with self.lock:
            for p in self.processes:
                if p.stdin and not p.stdin.closed:
                    p.stdin.close()
            for p in self.processes:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    p.wait(timeout=10)
                if p.stdout:
                    p.stdout.close()
            self.processes = []
            self.threads.shutdown(wait=True)
