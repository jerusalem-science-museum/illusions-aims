#!/usr/bin/env bash

echo "=== Mise à jour des paquets ==="
sudo apt update

echo "=== Vérification et installation des librairies ==="

install_if_missing () {
    PACKAGE=$1

    if dpkg -s "$PACKAGE" >/dev/null 2>&1; then
        echo "[OK] $PACKAGE déjà installé"
    else
        echo "[INSTALL] $PACKAGE"
        sudo apt install -y "$PACKAGE"
    fi
}

# Bibliothèques nécessaires

install_if_missing python3-qrcode
install_if_missing python3-serial
install_if_missing python3-opencv
install_if_missing python3-google-auth
install_if_missing python3-googleapi

echo
echo "=== Vérification Python ==="

python3 - <<EOF
modules = [
    "qrcode",
    "serial",
    "cv2",
    "google.auth",
    "googleapiclient"
]

for m in modules:
    try:
        __import__(m)
        print("[OK]", m)
    except:
        print("[MANQUANT]", m)
EOF

echo
echo "=== Terminé ==="