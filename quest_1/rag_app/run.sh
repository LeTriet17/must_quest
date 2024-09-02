# export PYTHONPATH=chatchat_server
#!/bin/bash

# Start Nginx
nginx -g 'daemon off;' &

export PYTHONPATH=chatchat_server

# Run your Python script
python chatchat_server/chatchat/startup.py --all-api
