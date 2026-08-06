# Runtime Profiles

Runtime profiles separate accelerator identity from service configuration.

## Naming

Accelerator profile identifiers use the following format:

```text
<vendor>.<accelerator-family>.<model>
```

Examples:

| Identifier | File |
|---|---|
| `nvidia.jetson.agx-orin` | `accelerators/nvidia/jetson/agx-orin.yaml` |
| `nvidia.rtx.a6000` | `accelerators/nvidia/rtx/a6000.yaml` |

The identifier is stable configuration vocabulary. Product names remain in profile
metadata and deployment documentation, not in vLLM argument branching.

## Directory structure

```text
config/profiles/
└── accelerators/
    └── <vendor>/
        └── <accelerator-family>/
            └── <model>.yaml
```

Each profile contains accelerator metadata and measured service defaults. The selected
deployment overlay can override individual fields, and local configuration remains the
highest-priority layer.

## Adding a profile

1. Add a YAML file under the vendor and accelerator-family directories.
2. Set the matching `accelerator.profile` identifier and complete accelerator metadata.
3. Define only measured service settings in the profile.
4. Select the profile through `accelerator.profile` in a deployment overlay.
5. Add configuration and engine-argument tests before target validation.

The loader applies the profile after `config/default.yaml` and before the selected
deployment overlay. `KOTONOHA__ACCELERATOR__PROFILE` selects a profile without modifying
a YAML file. Docker-specific settings for the same identifier live under
`docker/profiles/accelerators/`.
