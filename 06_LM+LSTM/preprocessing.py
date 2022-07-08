import re

START_TOKEN = '\x00'
NEWLINE_TOKEN = '\x01'
UNKNOWN_TOKEN = '\x02'
PARAGRAPH_TOKEN = '\x03'

TOKENS = [
  START_TOKEN,
  NEWLINE_TOKEN,
  UNKNOWN_TOKEN,
  PARAGRAPH_TOKEN,
]

def to_readable_text(text):
  text = str(text).replace(START_TOKEN, '\n\n\n' + '--' * 30 + '\n')
  text = text.replace(NEWLINE_TOKEN, '\n')
  text = text.replace(PARAGRAPH_TOKEN, '\n\n')
  text = text.replace(UNKNOWN_TOKEN, '<unknown>')
  return text


def read_file(path):
  with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
  t = re.sub('\n{4,}', f' {START_TOKEN}\n', text.lower())
  t = re.sub('\.{2,}', '. ', t)
  t = re.sub('\s{2,}', ' ', t).strip()
  t = re.sub('([!\"#$%\'[\]^_`{|}~])', r' \1 ', t)
  t = t.replace('\n{2,}', f' {PARAGRAPH_TOKEN}\n')
  t = t.replace('\n', f' {NEWLINE_TOKEN}\n')
  return t


def write_file(path, text):
  with open(path, 'w', encoding='utf-8') as f:
    f.write(to_readable_text(text))

