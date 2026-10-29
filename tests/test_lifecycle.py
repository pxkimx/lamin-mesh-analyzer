"""The app must not leave anything running after it is closed.  Run:  python -m pytest tests -q"""
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lifecycle  # noqa: E402



def free_port():
    """A port nobody is using right now (asked from the OS), so the test never collides with a running tool."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_gone(pid, seconds=10.0):
    end = time.time() + seconds
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def test_rule_for_quitting():
    step = lifecycle.should_quit
    state = (False, False, 0, False)
    for _ in range(500):                                   # nobody has opened the app: it waits
        state = step(0, *state[1:])
        assert not state[0]
    quit_now, seen, empty, closing = step(1, False, 0, False)
    for i in range(lifecycle.AFTER_CLOSE):                 # the page said it was closing
        quit_now, seen, empty, closing = step(0, seen, empty, True)
        assert quit_now == (i == lifecycle.AFTER_CLOSE - 1)
    assert step(1, True, 2, True) == (False, True, 0, False)   # a reload reconnects and cancels it
    empty = 0
    for i in range(lifecycle.AFTER_SILENCE):               # disconnected without a word
        quit_now, seen, empty, closing = step(0, True, empty, False)
        assert quit_now == (i == lifecycle.AFTER_SILENCE - 1)


def test_stopping_takes_started_processes_along():
    code = "import subprocess, time; subprocess.Popen(['sleep', '60']); print('up', flush=True); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert proc.stdout.readline().strip() == "up"
    group = subprocess.run(["pgrep", "-g", str(proc.pid)], capture_output=True, text=True).stdout.split()
    assert len(group) == 2
    lifecycle.register_child(proc)
    lifecycle.stop_children()
    proc.wait(timeout=10)
    assert all(wait_gone(int(pid)) for pid in group)


def test_server_quits_when_its_launcher_is_force_quit():
    TEST_PORT = free_port()
    launcher = subprocess.Popen(["sleep", "300"])          # stands in for the .app launcher
    env = dict(os.environ, **{lifecycle.LAUNCHER_ENV: str(launcher.pid)})
    server = subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(ROOT, "serve.py"),
                               "--port", str(TEST_PORT), "--headless"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.time() + 60
        while time.time() < end:
            try:
                if urllib.request.urlopen(f"http://localhost:{TEST_PORT}/_stcore/health", timeout=1).read() == b"ok":
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise AssertionError("server did not start")
        launcher.send_signal(signal.SIGKILL)               # a Dock force-quit: no chance to clean up
        launcher.wait()
        try:                                               # the server is our child: wait() also reaps it
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            raise AssertionError("server kept running after its launcher was gone")
        assert not port_open(TEST_PORT), "something still listens on the port"
    finally:
        for p in (launcher, server):
            if p.poll() is None:
                p.kill()
