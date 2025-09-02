#!/bin/bash

# Aller dans le dossier du script (au cas où)
cd "$(dirname "$0")"

echo "🚀 Démarrage du serveur RTSP + ngrok..."

# Lancer le script Python
python3 realtime.py
