# Configuration Architecture

## Storage Model

Kotonoha stores runtime configuration in `data/kotonoha.db`. The Web process and all
resident model services mount the same database. SQLite uses write-ahead logging, a
five-second busy timeout, and owner-only file permissions.

| Table | Key | Content |
|---|---|---|
| `configuration_defaults` | Profile identifier | Packaged base defaults and hardware-profile documents |
| `configuration_state` | `hardware_profile` | Selected hardware profile |
| `configuration_state` | `overrides` | Operator changes as a JSON object |
| `configuration_state` | `migration:*` | Completed legacy YAML imports |

Packaged YAML remains the source for version-controlled defaults. Startup synchronizes
those documents into `configuration_defaults` without replacing the selected profile or
operator overrides.

## Resolution Order

Settings resolve in the following order:

1. Base defaults from `config/default.yaml`
2. The selected document under `config/profiles/accelerators/`
3. Operator overrides stored in SQLite
4. `KOTONOHA__SECTION__FIELD` process environment values

Environment values are process-local and do not appear as SQLite overrides. Resetting a
field therefore cannot remove an environment override.

## Hardware Profiles

Hardware profiles define models, engine limits, service endpoints, and memory settings
for one deployment host. Identifiers use `<vendor>.<family>.<model>` and map to
`config/profiles/accelerators/<vendor>/<family>/<model>.yaml`.

`KOTONOHA_INITIAL_HARDWARE_PROFILE` seeds the selection only when the database does not
contain `hardware_profile`. A Web configuration save changes the persisted selection.
Resetting all settings preserves that selection and removes every operator override.
An explicit `scripts/manage.sh deploy jetson|a6000` operation selects the matching profile
before target validation.

## Web Operations

`GET /api/config` returns each editable field with its effective value, active default,
modified state, and available hardware profiles. `PUT /api/config` validates a complete
candidate before writing changes. `DELETE /api/config` removes selected overrides or all
overrides.

Changes take effect without restarting the Web process. Existing browser sessions receive
the new settings, and model-owned changes invoke the affected service `/admin/reload`
endpoint. A reload failure is reported without rolling back the validated database write.

## Legacy Migration

The loader imports files selected by `--config`, `KOTONOHA_CONFIG`,
`KOTONOHA_LOCAL_CONFIG`, and an existing `config/local.yaml` once. Each absolute source
path receives a migration marker. Later edits to those YAML files do not change runtime
configuration.

Remove legacy files after confirming the imported values in the Web configuration page.
Do not delete `data/kotonoha.db`; it also stores glossary and history data.
