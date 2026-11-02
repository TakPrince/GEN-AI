# REMOTE_SSH — Ubuntu Headless Runbook (Asus Laptop)

Goal: SSH into the laptop, bind firewall rules, and disable lid-close sleep so the
guardrail dashboard / eval runner can run remotely unattended.

## 1. Install and enable OpenSSH server

```bash
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable --now ssh
systemctl status ssh --no-pager
sudo ufw allow OpenSSH
sudo ufw enable
sudo ufw status verbose
```

- Find the target IP: `ip a show` or `hostname -I`.
- Reachability check from the client: `ssh <user>@<ip>`.

## 2. Bind network / firewall rules

```bash
# Dashboards and eval UI
sudo ufw allow 8501/tcp comment 'streamlit'
# Restrict SSH to the LAN subnet only (edit to match your network)
sudo ufw allow from 192.168.1.0/24 to any port 22 proto tcp comment 'ssh-lan'
sudo ufw reload
sudo ufw status numbered
```

- Verify a port is free before launching the UI: `ss -tlnp | grep 8501`.
- Optional static IP (Netplan, Ubuntu 24.04):

```bash
sudo nano /etc/netplan/01-lan.yaml
# network:
#   version: 2
#   ethernets:
#     enp3s0:            # interface name from `ip a`
#       addresses: [192.168.1.50/24]
#       routes: [{to: default, via: 192.168.1.1}]
#       nameservers: {addresses: [192.168.1.1]}
sudo chmod 600 /etc/netplan/01-lan.yaml
sudo netplan apply
```

## 3. Block lid-close sleep (systemd power masks)

```bash
# Mask every power target that can suspend the machine
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

- Tell logind to ignore lid events:

```bash
sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
sudo sed -i 's/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf
sudo systemctl restart systemd-logind
```

- Verify:

```bash
systemctl status sleep.target suspend.target --no-pager   # loaded: masked
systemd-inhibit --list                                    # blocking inhibitors visible
grep HandleLidSwitch /etc/systemd/logind.conf              # HandleLidSwitch=ignore
```

- Undo (re-enable suspend):

```bash
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo sed -i 's/^HandleLidSwitch=.*/#HandleLidSwitch=suspend/' /etc/systemd/logind.conf
sudo systemctl restart systemd-logind
```

## 4. Persist a running session (tmux)

```bash
sudo apt install -y tmux
tmux new -s guardrails
make ui        # streamlit dashboard on 0.0.0.0:8501
# detach: Ctrl-b d  |  reattach: tmux attach -t guardrails
```

- After reboot the masked power state persists automatically; no re-run needed.
- Quick sanity: `uptime -p` (no suspend gaps), `journalctl -u systemd-logind -b | grep -i lid`.
