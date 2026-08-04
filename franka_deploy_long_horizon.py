#!/usr/bin/env python3
"""Run the single-cycle gear policy as a three-cycle long-horizon demo.

Robot motion, policy inference, and camera capture reuse
``franka_deploy_gear.py``. Press W when the current extraction cycle
is complete. The robot opens its gripper, returns to sim-home, and starts the
next policy cycle automatically.

Example:
    python franka_deploy_long_horizon.py \
        -c GeoBridgeCheckpoints/FPSA_gear/latest.ckpt \
        -o data/FPSA_gear_long_horizon \
        --camera_setup both

Controls:
    S      Start the full long-horizon episode
    W      Mark the current cycle complete
    P      Pause / resume
    H      End the entire long-horizon episode immediately
    Esc    Quit
"""
import sys

from franka_deploy_gear import main as deploy_main


def _has_option(args, name):
    return any(arg == name or arg.startswith(f'{name}=') for arg in args)


if __name__ == '__main__':
    argv = sys.argv[1:]
    defaults = []
    if not _has_option(argv, '--long-horizon-cycles'):
        defaults.extend(['--long-horizon-cycles', '3'])
    deploy_main.main(
        args=defaults + argv,
        prog_name='franka_deploy_long_horizon.py',
    )
