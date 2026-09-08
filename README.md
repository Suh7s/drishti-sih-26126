# DRISHTI

Camera-first navigation prototype for SIH problem 26126.

Start with **START_HERE.md**. For the next Ubuntu Codex session, use
**UBUNTU_HANDOFF.md**. Evidence and limitations are in **docs/STATUS.md**.

Target simulation: Ubuntu + RTX 5080 + Isaac Sim 6.0.1.
Current status: portable prototype and experiments implemented; Isaac/Spot
integration and cinematic recording require runtime debugging and verification.

```bash
python3 run.py doctor
python3 run.py test
python3 run.py demo
```

Portable development dependencies are in requirements.txt. Use a separate virtual
environment for portable development. Isaac Sim uses its own Python environment.

The offline demo is a 2-D synthetic-observation replay, not an Isaac recording.
