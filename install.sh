#!/usr/bin/env bash
python3 -m venv venv
source venv/bin/activate
pip install update pip
pip install wheel
pip install -r requirements.txt
pre-commit install

echo '#!/usr/bin/env bash' > ~/capture-login.sh
echo $PWD/venv/bin/python $PWD/detcap.py >> ~/capture-login.sh
chmod +x ~/capture-login.sh

echo '#!/usr/bin/env bash' > ~/recognize-login.sh
echo $PWD/venv/bin/python $PWD/recognize_me.py >> ~/recognize-login.sh
chmod +x ~/recognize-login.sh
