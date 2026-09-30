import datetime
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import sleep

from openai import OpenAI
import shutil

class VLLMRunner:
    """
    Run a vLLM server either:
      - in a container (Docker/Podman), or
      - directly in the current environment (no container) if use_container=False.

    It launches the server as a subprocess in its own process group. Prefer stopping
    with SIGINT (Ctrl+C) to allow graceful shutdown.
    """

    def __init__(
        self,
        model_name: str,
        logfile: str | None = None,
        port: int = 8000,
        max_model_length: int = 512,
        tensor_parallel_size: int = 1,
        trials: int = 40,
        initial_sleep: int = 30,
        sleep_interval: int = 30,
        use_container: bool = False,  # default: run vLLM directly (no nested containers)
        container_image: str = "ghcr.io/lambdalabsml/vllm-builder:v0.10.0-cu128-ubuntu22.04-arm64",
        gpu_memory_utilization: float = 0.6,
    ) -> None:
        self.model_name = model_name
        self.port = port
        self.max_model_length = max_model_length
        self.tensor_parallel_size = tensor_parallel_size
        self.trials = trials
        self.initial_sleep = initial_sleep
        self.sleep_interval = sleep_interval
        self.logfile = logfile
        self._logfile_handle = None
        self.pid = None
        self.proc = None
        self.use_container = use_container
        self.container_image = container_image
        self.gpu_memory_utilization = gpu_memory_utilization

        self.test_client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{self.port}/v1",
            timeout=600,
        )

        self._logfiles_base_dir = Path(__file__).parent.parent.parent / "vllm_logs"
        self._logfiles_base_dir.mkdir(parents=True, exist_ok=True)
        if self.logfile is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = self.model_name.replace("/", "_").replace(":", "_").replace("-", "_")
            self.logfile = str(self._logfiles_base_dir / f"vllm_{safe_model_name}_{timestamp}.log")

        # Determine if model_name refers to a local path (directory or file)
        self.is_local_model = os.path.exists(model_name)
        self.container_name = "vllm_container"

        # Container engine (if used)
        self.engine = (
            os.environ.get("CONTAINER_ENGINE")
            or shutil.which("docker")
            or shutil.which("podman")
            or "podman"
        )

        # Normalize local path if needed
        if self.is_local_model:
            self.model_path_abs = os.path.abspath(self.model_name)
        else:
            self.model_path_abs = self.model_name  # likely a HF repo id

        self.served_model_name = None

    def _check_online(self) -> bool:
        try:
            r = self.test_client.chat.completions.create(
                model=self.served_model_name,
                n=1,
                messages=[{"role": "user", "content": "Hello"}],
                timeout=5,
                max_tokens=2,
            )
            text = r.choices[0].message.content
            return bool(text)
        except Exception as e:
            print(f"Error at _check_online: {e}")
            return False

    def _vllm_server_args(self):
        return [
            "--host", "0.0.0.0",
            "--port", str(self.port),
            # "--max-model-len", str(self.max_model_length),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

    def __enter__(self) -> "VLLMRunner":
        def handle_sigint(signum, frame):
            print(f"Received signal {signum}, shutting down vLLM server...")
            self.__exit__(None, None, None)
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)
        self._logfile_handle = open(self.logfile, "a")

        if self.use_container:
            # Container mode
            if self.is_local_model:
                container_model_path = os.path.join("/model", os.path.basename(self.model_path_abs))
                volume_mount = ["-v", f"{self.model_path_abs}:{container_model_path}"]
                self.served_model_name = container_model_path
            else:
                volume_mount = []
                self.served_model_name = self.model_name

            _cmd_run = [
                self.engine, "run",
                "--name", self.container_name,
                "--gpus", "all",
                "-p", f"{self.port}:{self.port}",
                "--rm",
            ]
            _cmd_image_and_model = [
                self.container_image,
                self.served_model_name,
            ] + self._vllm_server_args()

            cmd = _cmd_run + volume_mount + _cmd_image_and_model

        else:
            # Non-container mode: explicitly start the vLLM server
            # Try "vllm serve" first, fall back to "python -m vllm.entrypoints.openai.api_server"
            if self.is_local_model:
                self.served_model_name = self.model_path_abs
            else:
                self.served_model_name = self.model_name

            vllm_bin = shutil.which("vllm")
            if vllm_bin:
                cmd = ["vllm", "serve", self.served_model_name] + self._vllm_server_args()
            else:
                # Fallback to module invocation
                cmd = [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", self.served_model_name,
                ] + self._vllm_server_args()

        print(f"Starting vLLM with command: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._logfile_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            preexec_fn=os.setsid,
        )
        self.pid = self.proc.pid
        sleep(self.initial_sleep)

        if self.proc.poll() is not None:
            self.__exit__(None, None, None)
            raise Exception(
                f"vLLM process with PID {self.pid} terminated unexpectedly, see log at {self.logfile} for details"
            )

        for trial in range(self.trials):
            print(f"Trial {trial + 1}/{self.trials}: Checking if vLLM is online...")
            if self._check_online():
                print(f"vLLM is online after {trial * self.sleep_interval + self.initial_sleep} seconds")
                return self
            else:
                print(f"vLLM not online yet, waiting {self.sleep_interval} seconds (trial {trial + 1}/{self.trials})...")
                sleep(self.sleep_interval)

        self.__exit__(None, None, None)
        raise Exception(f"vLLM failed to start in time, see log at {self.logfile} for details")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if getattr(self, "_exit_called", False):
            return
        self._exit_called = True

        if self.proc and self.proc.poll() is None:
            print(f"Terminating vLLM process with PID {self.pid}")
            try:
                os.killpg(os.getpgid(self.pid), signal.SIGINT)
                self.proc.wait()
                assert self.proc.poll() is not None
                print(f"vLLM process with PID {self.pid} terminated.")
            except Exception as e:
                print(f"Failed to terminate vLLM process at PID: {self.pid} gracefully: {e}.")
                try:
                    print(f"Killing vLLM process with PID {self.pid}")
                    os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                    self.proc.wait()
                    assert self.proc.poll() is not None
                    print(f"vLLM process with PID {self.pid} killed")
                except Exception as e:
                    print(f"Failed to kill vLLM process at PID: {self.pid}: {e}. You might need to kill it manually.")

            if self.use_container:
                print(f"Terminating {self.engine} container {self.container_name}...")
                try:
                    subprocess.run(
                        [self.engine, "stop", self.container_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=self._logfile_handle,
                    )
                except Exception as e:
                    print(f"Failed to stop container {self.container_name}: {e}. You might need to stop it manually.")

        if self._logfile_handle:
            self._logfile_handle.close()
            print(f"vLLM log file saved at: {self.logfile}")

