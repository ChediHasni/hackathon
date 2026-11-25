import sys, unicodedata
sys.path.insert(0, 'c:/Users/User/Desktop/hackathon/hackathon/zonescorer/backend')
from api.location_resolver import resolve_location, _norm

text = 'اريد حي هادئ وأخضر في مرسى'
print('Norm text:', _norm(text))
r = resolve_location(text, limit=5, min_score=0.3)
print('Results:', len(r))
for x in r:
    print(' ', x['name'], x['score'], x['match_type'])