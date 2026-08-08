# Hardware Profiles

Hardware profiles bind host identity, accelerator metadata, model selection, and measured
runtime limits to one stable identifier.

## Naming

Hardware profile identifiers use the following format:

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

Each profile contains accelerator metadata and measured service defaults. Startup
synchronizes every profile into SQLite. Operator changes remain separate overrides in
`data/kotonoha.db`.

## Adding a profile

1. Add a YAML file under the vendor and accelerator-family directories.
2. Set the matching `accelerator.profile` identifier and complete accelerator metadata.
3. Define only measured service settings in the profile.
4. Select the profile through the Web configuration page or the deployment script.
5. Add configuration and engine-argument tests before target validation.

The loader combines `config/default.yaml`, the selected hardware profile, and SQLite
operator overrides. `KOTONOHA__ACCELERATOR__PROFILE` is a non-persistent process override
for diagnostics. Docker-specific settings for the same identifier live under
`docker/profiles/accelerators/`.
