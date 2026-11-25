p = r'c:/Users/User/Desktop/hackathon/hackathon/zonescorer/frontend/templates/index.html'
with open(p, 'r', encoding='utf-8', errors='replace', newline='') as f:
    content = f.read()

old = """function getCsrf() {
  const m = document.cookie.match(/csrftoken=([^;]+)/);
  return m ? m[1] : '';
}
</script>"""

with open(r'c:/Users/User/Desktop/hackathon/hackathon/zonescorer/frontend/chat_js_part1.js', 'r', encoding='utf-8') as f:
    part1 = f.read()
with open(r'c:/Users/User/Desktop/hackathon/hackathon/zonescorer/frontend/chat_js_part2.js', 'r', encoding='utf-8') as f:
    part2 = f.read()

new = """function getCsrf() {
  const m = document.cookie.match(/csrftoken=([^;]+)/);
  return m ? m[1] : '';
}""" + part1 + part2 + "\n</script>"

if old not in content:
    print('ERROR: target text not found')
else:
    content = content.replace(old, new, 1)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    print('Chat JS injected successfully')