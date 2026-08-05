#!/usr/bin/env bash
# Run a hardware probe with the vendor image user, then return outputs to the host user.
set -uo pipefail

output_paths=()
previous_argument=""
for argument in "$@"; do
  case "$previous_argument" in
    --out|--md|--patch) output_paths+=("$argument") ;;
  esac
  previous_argument=$argument
done

for output_path in "${output_paths[@]}"; do
  output_directory=$(dirname "$output_path")
  if [ -d "$output_directory" ]; then
    chown "${SPIKE_USER_ID}:${SPIKE_GROUP_ID}" "$output_directory"
  fi
done

"$@"
command_status=$?

for output_path in "${output_paths[@]}"; do
  if [ -e "$output_path" ]; then
    chown "${SPIKE_USER_ID}:${SPIKE_GROUP_ID}" "$output_path"
  fi
done

exit "$command_status"
