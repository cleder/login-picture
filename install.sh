#!/usr/bin/env bash
python3 -m venv venv
source venv/bin/activate
pip install update pip
pip install wheel
pip install opencv-python

echo '#!/usr/bin/env bash' > ~/capture-login.sh
echo $PWD/venv/bin/python $PWD/detcap.py >> ~/capture-login.sh
chmod +x ~/capture-login.sh
