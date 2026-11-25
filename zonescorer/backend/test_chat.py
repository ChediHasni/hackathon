import os, sys, json
sys.path.insert(0, 'c:/Users/User/Desktop/hackathon/hackathon/zonescorer/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'zonescorer.settings')
import django
django.setup()
from django.test import Client
from django.test.utils import setup_test_environment
setup_test_environment()
client = Client()

# Test 1: greeting -> follow-up
resp = client.post('/api/chat/', data=json.dumps({'message': 'Bonjour'}), content_type='application/json')
print('Test 1 (greeting):', resp.status_code)
d = resp.json()
print('  Reply:', d.get('reply'))
print('  Follow-up count:', d.get('profile', {}).get('follow_up_count'))
print()

# Test 2: complete request
resp = client.post('/api/chat/', data=json.dumps({'message': 'Je veux un quartier vert et calme à La Marsa', 'session_id': 's1'}), content_type='application/json')
print('Test 2 (La Marsa green):', resp.status_code)
d = resp.json()
print('  Reply:', d.get('reply'))
print('  Zones count:', len(d.get('zones', [])))
print('  Profile:', json.dumps(d.get('profile', {}), ensure_ascii=False))
for z in d.get('zones', [])[:3]:
    print('  - %s: match=%s, strongest=%s, weakest=%s' % (z['zone']['name'], z['match_score'], z['strongest'], z['weakest']))
    print('    reasons:', z.get('reasons', []))
print()

# Test 3: Arabic request
resp = client.post('/api/chat/', data=json.dumps({'message': 'اريد حي هادئ وأخضر في مرسى', 'session_id': 's2'}), content_type='application/json')
print('Test 3 (Arabic):', resp.status_code)
d = resp.json()
print('  Reply:', d.get('reply'))
print('  Zones count:', len(d.get('zones', [])))
print('  Profile:', json.dumps(d.get('profile', {}), ensure_ascii=False))

print()
print('ALL TESTS PASSED')