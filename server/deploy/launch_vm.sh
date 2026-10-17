#!/usr/bin/env bash
# Create the Always Free VM, retrying while Oracle has no free ARM capacity:
#   server/deploy/launch_vm.sh
# Needs the oci CLI configured (~/.oci/config) and the wizard-made VCN
# "aslcv-vcn" (or SUBNET_ID=<ocid>). Launches exactly one
# VM.Standard.A1.Flex with 2 OCPUs / 12 GB (the whole free allowance) on
# Ubuntu 24.04, cycling through the availability domains every RETRY_SECONDS
# until one has capacity, then waits for it to run and prints the public IP.
# Checks for an existing instance before every launch, so it never creates a
# second one; run it again after an interruption and it picks up the first.
# SSH_PUB=~/.ssh/<key>.pub picks the key the VM accepts.
set -euo pipefail
NAME=aslcv-server
SHAPE=VM.Standard.A1.Flex
SHAPE_CONFIG='{"ocpus": 2, "memoryInGBs": 12}'
SSH_PUB=${SSH_PUB:-$HOME/.ssh/id_ed25519_personal.pub}
RETRY_SECONDS=${RETRY_SECONDS:-60}

export SUPPRESS_LABEL_WARNING=True
tenancy=$(awk -F= '/^tenancy/ {gsub(/ /, "", $2); print $2; exit}' ~/.oci/config)
log() { echo "$(date '+%F %T') $*"; }

# oci, retried while a newly added API key is still propagating (Oracle then
# answers NotAuthenticated on some calls and not others).
o() {
  local out err
  for _ in $(seq 20); do
    err=$(mktemp)
    if out=$(oci "$@" 2>"$err"); then rm -f "$err"; echo "$out"; return 0; fi
    if ! grep -q NotAuthenticated "$err"; then cat "$err" >&2; rm -f "$err"; return 1; fi
    rm -f "$err"; sleep 15
  done
  echo "oci: still NotAuthenticated after 5 minutes" >&2; return 1
}

# The first instance that isn't terminated or terminating, as "id state"
# (empty when none).
live_instance() {
  o compute instance list -c "$tenancy" --all \
    --query 'data[?!contains([`TERMINATED`, `TERMINATING`], "lifecycle-state")].[id, "lifecycle-state"] | [0]' \
    --raw-output | tr -d '[]",' | awk 'NF' | paste -sd' '
}

# Waits for an existing instance: prints its IP and exits once it runs, or
# returns if it terminates (a launch Oracle accepted, then ran out of room for).
wait_for() {
  local id=$1 state
  while true; do
    state=$(o compute instance get --instance-id "$id" --query 'data."lifecycle-state"' --raw-output)
    case $state in
      RUNNING)
        log "running after $attempt attempt(s)"
        echo "PUBLIC_IP=$(o compute instance list-vnics --instance-id "$id" --query 'data[0]."public-ip"' --raw-output)"
        exit 0 ;;
      TERMINATING|TERMINATED) log "instance $state, retrying"; return ;;
      *) sleep 10 ;;
    esac
  done
}

if [[ -z "${SUBNET_ID:-}" ]]; then
  vcn=$(o network vcn list -c "$tenancy" --display-name aslcv-vcn \
    --query 'data[0].id' --raw-output)
  [[ -n "$vcn" && "$vcn" != "null" ]] || { echo "no VCN named aslcv-vcn; create it with the VCN wizard or set SUBNET_ID"; exit 1; }
  SUBNET_ID=$(o network subnet list -c "$tenancy" --vcn-id "$vcn" \
    --query 'data[?"prohibit-public-ip-on-vnic" == `false`] | [0].id' --raw-output)
  [[ -n "$SUBNET_ID" && "$SUBNET_ID" != "null" ]] || { echo "aslcv-vcn has no public subnet"; exit 1; }
fi

image=$(o compute image list -c "$tenancy" --operating-system "Canonical Ubuntu" \
  --operating-system-version 24.04 --shape "$SHAPE" --sort-by TIMECREATED --sort-order DESC \
  --query 'data[?!contains("display-name", `Minimal`)] | [0].id' --raw-output)
[[ -n "$image" && "$image" != "null" ]] || { echo "no Ubuntu 24.04 image for $SHAPE"; exit 1; }

mapfile -t ads < <(o iam availability-domain list -c "$tenancy" --query 'data[].name' --raw-output | tr -d '[]", ' | grep .)
log "subnet $SUBNET_ID, image $image, ${#ads[@]} availability domain(s)"

attempt=0
while true; do
  for ad in "${ads[@]}"; do
    attempt=$((attempt + 1))
    # Re-checked before every launch: a launch that succeeded but whose reply
    # was lost must never be followed by a second one.
    live=$(live_instance)
    if [[ -n "$live" ]]; then
      log "instance exists (${live#* }), waiting for it"
      wait_for "${live%% *}"
      continue
    fi
    err=$(mktemp)
    if oci compute instance launch -c "$tenancy" --availability-domain "$ad" \
        --display-name "$NAME" --shape "$SHAPE" --shape-config "$SHAPE_CONFIG" \
        --image-id "$image" --subnet-id "$SUBNET_ID" --assign-public-ip true \
        --ssh-authorized-keys-file "$SSH_PUB" >/dev/null 2>"$err"; then
      log "attempt $attempt: launch accepted in $ad"
      rm -f "$err"; continue
    fi
    if grep -qi "capacity" "$err"; then
      log "attempt $attempt: $ad out of capacity"
    elif grep -q "TooManyRequests" "$err"; then
      log "attempt $attempt: rate limited, backing off"; sleep $((RETRY_SECONDS * 3))
    elif grep -q "NotAuthenticated" "$err"; then
      log "attempt $attempt: API key not accepted yet, retrying"; sleep 15
    else
      log "attempt $attempt: unexpected error, stopping:"; cat "$err"; rm -f "$err"; exit 1
    fi
    rm -f "$err"
  done
  sleep "$RETRY_SECONDS"
done
