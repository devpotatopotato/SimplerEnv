# Policy servers

Each model runs in an isolated Python environment and exposes the same small HTTP/JSON API. Only `simpler_protocol`, NumPy, and that model's own upstream dependencies are needed on the server side; SAPIEN is never imported there.

| Adapter | Canonical adapted mode | Released diagnostic mode |
| --- | --- | --- |
| π0.5 | Physical Cartesian 7-D output | `libero_normalized`, with explicit scales/convention |
| VPP | Physical Cartesian 7-D output | `calvin_normalized`, converted by CALVIN's 50/20 factors |
| Cosmos 3 Edge | Upstream `midtrain` Cartesian output converted from absolute poses | None; public DROID joint-position output is rejected |

Run any module with `--help` for model-specific checkpoint arguments:

```bash
python -m policy_servers.pi05.server --help
python -m policy_servers.vpp.server --help
python -m policy_servers.cosmos3.server --help
```

The adapters are intentionally thin. Clone and install the official [OpenPI](https://github.com/Physical-Intelligence/openpi), [VPP](https://github.com/roboterax/video-prediction-policy), or [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) repository in a separate environment, then install this repository with `pip install -e /path/to/SimplerEnv --no-deps` to add the wire protocol and adapter.
