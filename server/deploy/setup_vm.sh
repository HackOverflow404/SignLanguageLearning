#!/usr/bin/env bash
# One-time setup on a fresh Oracle Cloud Ubuntu VM (run by deploy.sh --setup):
# Docker + Compose, and ports 80/443 opened in the VM's own firewall. Oracle's
# Ubuntu images ship iptables rules that reject everything but SSH, on top of
# the cloud-side security list (opened separately in the console).
set -euo pipefail
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 iptables-persistent
sudo usermod -aG docker "$USER"
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport $port -m state --state NEW -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 6 -p tcp --dport $port -m state --state NEW -j ACCEPT
done
sudo netfilter-persistent save
echo "VM ready: docker $(docker --version | cut -d' ' -f3 | tr -d ,), ports 80/443 open"
