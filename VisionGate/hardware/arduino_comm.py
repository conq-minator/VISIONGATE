"""
hardware/arduino_comm.py
========================
Hardware abstraction layer for gate and buzzer control.

In SIMULATION_MODE (default), all commands are printed to console and logged.
To use real Arduino hardware:
  1. Set SIMULATION_MODE = False in config.py
  2. Connect Arduino via USB
  3. Set ARDUINO_PORT to the correct COM port
  4. Flash the gate sketch to Arduino

The class is designed so that only this file needs to change when moving
from simulation to real hardware.
"""

import time
import logging
import threading

import config

logger = logging.getLogger(__name__)

# Try to import pyserial for real hardware; not required in simulation mode
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class ArduinoComm:
    """
    Sends gate and buzzer commands via serial (or simulates them).

    Thread-safe: all public methods acquire a mutex before sending.
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self._serial     = None
        self._sim_mode   = config.SIMULATION_MODE

        if not self._sim_mode:
            self._connect()

    # ── Public commands ───────────────────────────────────────────────────────

    def open_gate(self):
        """Open the access gate."""
        self.send_command("OPEN_GATE")
        # Auto-close after configured duration
        threading.Timer(config.GATE_OPEN_DURATION, self._close_gate).start()

    def buzzer_on(self):
        """Activate the denial buzzer."""
        self.send_command("BUZZER_ON")

    def send_command(self, command: str):
        """
        Send an arbitrary command string.
        In simulation mode, prints to console and logs.
        In real mode, sends over serial.
        """
        with self._lock:
            if self._sim_mode:
                self._simulate(command)
            else:
                self._send_serial(command)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _close_gate(self):
        with self._lock:
            if self._sim_mode:
                self._simulate("CLOSE_GATE")
            else:
                self._send_serial("CLOSE_GATE")

    def _simulate(self, command: str):
        ts = time.strftime("%H:%M:%S")
        msg = f"[{ts}] [HW SIM] >>> {command}"
        print(msg)
        logger.info(f"[HW SIM] {command}")

    def _connect(self):
        if not SERIAL_AVAILABLE:
            logger.error(
                "pyserial not installed. Cannot connect to Arduino. "
                "Falling back to simulation mode."
            )
            self._sim_mode = True
            return

        try:
            self._serial = serial.Serial(
                port=config.ARDUINO_PORT,
                baudrate=config.ARDUINO_BAUD_RATE,
                timeout=1,
            )
            time.sleep(2)  # Allow Arduino to reset
            logger.info(
                f"Arduino connected on {config.ARDUINO_PORT} "
                f"@ {config.ARDUINO_BAUD_RATE} baud."
            )
        except Exception as exc:
            logger.error(
                f"Failed to connect to Arduino on {config.ARDUINO_PORT}: {exc}. "
                "Falling back to simulation mode."
            )
            self._sim_mode = True

    def _send_serial(self, command: str):
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(f"{command}\n".encode("utf-8"))
                logger.info(f"[HW SERIAL] {command}")
            except Exception as exc:
                logger.error(f"Serial write error: {exc}")
        else:
            logger.warning(f"Serial port not open – simulating: {command}")
            self._simulate(command)

    def close(self):
        """Release serial port on shutdown."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("Arduino serial port closed.")
