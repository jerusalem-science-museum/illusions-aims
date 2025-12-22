#!/usr/bin/env bash

source ~/Public/infra/scripts/config_rpi.sh

safe_append_config "gpu_mem=256"

sudo sh -c "cat > /lib/systemd/system/sms_camera.service <<EOF
[Unit]
Description=SMS Camera
After=multi-user.target
[Service]
Type=simple
ExecStart=/bin/bash /home/pi/Public/sms_camera/run_camera_gsm_to_url.sh --interface rpyc
Restart=on-abort
[Install]
WantedBy=multi-user.target
EOF"

sudo chmod 644 /lib/systemd/system/sms_camera.service
sudo systemctl daemon-reload
sudo systemctl enable sms_camera.service
sudo systemctl start sms_camera.service
sudo mkdir /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
sudo sh -c "cat >> /etc/systemd/journald.conf <<EOF
SystemMaxUse=10M
EOF"
sudo systemctl restart systemd-journald

sudo sh -c "cat >> /home/pi/.bash_aliases <<EOF
alias sms_camera_log='sudo journalctl -e -f -n 50 -o cat -u sms_camera.service'
alias sms_camera_log_all='sudo journalctl --no-tail --no-pager -m -o cat -u sms_camera.service'
alias sms_camera_restart='sudo systemctl restart sms_camera.service'
alias sms_camera_stop='sudo systemctl stop sms_camera.service'
alias sms_camera_client='/home/pi/Public/infra/scripts/run_client.sh'
alias sms_camera_take_picture='/home/pi/Public/infra/scripts/run_client.sh --cmd "app.take_picture()"'
alias sms_camera_usb_restart='sudo systemctl stop sms_camera.service; sleep 2; for i in "unbind" "bind"; do sudo sh -c "echo 1-1 > /sys/bus/usb/drivers/usb/$i" && sleep 8; done; sleep 10; sudo systemctl start sms_camera.service'
alias sms_camera_mail_picture='raspistill -n -o /tmp/camera_picture.png && mpack -s "SMS Camera" /tmp/camera_picture.png'
alias sms_camera_mail_video='raspivid -w 640 -h 480 -t 2000 -o /tmp/camera_video.h264 && MP4Box -add /tmp/camera_video.h264 /tmp/camera_video.mp4 && mpack -s "SMS Camera" /tmp/camera_video.mp4'
alias sms_camera_gdrive='gdrive -c /home/pi/Public/keys/old/ --service-account logger-995ad2d4b91d.json'
alias google_drive='gdrive -c /home/pi/Public/keys/ --service-account google_service_account.json'
EOF"

sudo sh -c 'cat > /etc/udev/rules.d/90-usb-serial.rules <<EOF
SUBSYSTEM=="tty",KERNELS=="1-1.2:1.0",SYMLINK+="ttyGsmUart"
EOF'

sudo sh -c "cat > /etc/ssmtp/ssmtp.conf <<EOF
root=<email>
mailhub=smtp.gmail.com:587
FromLineOverride=YES
AuthUser=<email>
AuthPass=<password>
UseSTARTTLS=YES
UseTLS=YES
EOF"

sudo install /home/pi/Public/infra/bin/gdrive/linux_arm/gdrive /usr/local/bin/gdrive

gdrive_command share --role writer --type user --email <email> <28 digit drive folder id>

# SIM800:
# profile save and param save
# > ATE0;+CMGF=1;+CNMI=2,2,0,0,0;+CSCS="UCS2";+CSMP=17,167,0,8;+CSAS;+IPR=115200;&W

# Air200:
# profile save and param save
# > ATE0;+CMGF=1;+CNMI=2,2,0,0,0;+CSCS="UCS2";+CSMP=17,167,0,8;+CSAS;+IPR=115200
# > AT&W
