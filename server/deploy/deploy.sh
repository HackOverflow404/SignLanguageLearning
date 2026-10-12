#!/usr/bin/env bash
# Deploy the sentence server to the Oracle VM:
#   server/deploy/deploy.sh ubuntu@<public-ip> [--setup]
# --setup (first time only) installs Docker and opens the firewall. Then it
# copies the build context + compose files to ~/aslcv-server, builds the image
# ON the VM (native ARM), starts it, and checks /health over HTTPS. The public
# address is the IP as an sslip.io name, e.g. 129-146-1-2.sslip.io.
# SSH_KEY=~/.ssh/<key> picks a key other than ssh's default.
set -euo pipefail
host=${1:?usage: deploy.sh user@ip [--setup]}
here=$(cd "$(dirname "$0")" && pwd)
ip=${host#*@}
site="${ip//./-}.sslip.io"
ssh_opts=(-o StrictHostKeyChecking=accept-new ${SSH_KEY:+-i "$SSH_KEY"} ${SSH_KEY:+-o IdentitiesOnly=yes})

if [[ "${2:-}" == "--setup" ]]; then
  ssh "${ssh_opts[@]}" "$host" 'bash -s' < "$here/setup_vm.sh"
fi
"$here/../build.sh"
rsync -az --delete -e "ssh ${ssh_opts[*]}" "$here/../build/app/" "$host:aslcv-server/app/"
rsync -az -e "ssh ${ssh_opts[*]}" "$here/compose.yaml" "$here/Caddyfile" "$host:aslcv-server/"
ssh "${ssh_opts[@]}" "$host" "cd aslcv-server && echo SITE_ADDRESS=$site > .env && sg docker -c 'docker compose up -d --build'"
echo "waiting for https://$site/health ..."
for _ in $(seq 60); do
  curl -sf "https://$site/health" && { echo; echo "live: https://$site"; exit 0; }
  sleep 5
done
echo "not answering yet; check: ssh $host 'cd aslcv-server && docker compose logs --tail 50'"
exit 1
