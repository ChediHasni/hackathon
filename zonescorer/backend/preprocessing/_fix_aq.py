import re

with open('air_quality.py', 'rb') as f:
    content = f.read().decode('utf-8')

old = (
    "    gee_service_account = config('GEE_SERVICE_ACCOUNT', default='')\r\n"
    "    gee_key_file = config('GEE_KEY_FILE', default='')\r\n"
    "    credentials = ee.ServiceAccountCredentials(gee_service_account, gee_key_file)\r\n"
    "    ee.Initialize(credentials)\r\n"
)

new = (
    "    gee_service_account = config('GEE_SERVICE_ACCOUNT', default='')\r\n"
    "    gee_key_file = config('GEE_KEY_FILE', default='')\r\n"
    "    gee_project = config('GEE_PROJECT', default='zonescorer-494214')\r\n"
    "    credentials = ee.ServiceAccountCredentials(gee_service_account, gee_key_file)\r\n"
    "    ee.Initialize(credentials, project=gee_project)\r\n"
    "    print(f'[air_quality] Authenticated with project: {gee_project}')\r\n"
)

if old in content:
    result = content.replace(old, new, 1)
    with open('air_quality.py', 'wb') as f:
        f.write(result.encode('utf-8'))
    print('SUCCESS: air_quality.py updated')
else:
    print('NOT FOUND')
    idx = content.find('gee_service_account')
    print(repr(content[idx:idx+300]))
