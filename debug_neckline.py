from src.datasets.attribute_extractor import AttributeExtractor

e = AttributeExtractor()

# Test the neckline detection issue from Index 2
text = 'The lady is wearing a short-sleeve shirt with graphic patterns. The shirt is with cotton fabric. The neckline of the shirt is round. There is an accessory on her wrist. This woman wears a ring. There is an accessory in his her neck.'
print('Text:', text)
print()

# Split into sentences
sentences = e._split_into_sentences(text)
print('Sentences:')
for i, s in enumerate(sentences):
    print(f'  {i}: {s}')
print()

# Test extraction
result = e.extract(text, 'tees')
print('Extraction result:', result)
print()

# Test individual sentences
for i, sentence in enumerate(sentences):
    if 'neckline' in sentence.lower():
        print(f'Sentence {i} contains neckline:')
        print(f'  \"{sentence}\"')
        neckline_matches = e._extract_from_sentence(sentence, 'upper')
        print(f'  Neckline matches: {neckline_matches}')
        print()

# Test the pattern matching directly
sentence = 'The neckline of the shirt is round.'
print(f'Direct pattern test for: "{sentence}"')
print('Round triggers:', e.get_all_triggers('neckline', 'round'))
print('All neckline triggers:', e.get_all_triggers('neckline', 'round') + e.get_all_triggers('neckline', 'crew') + e.get_all_triggers('neckline', 'v_neck'))

# Check regex matching
import re
for trigger in e.get_all_triggers('neckline', 'round'):
    pattern = r'\b' + re.escape(trigger.lower()) + r'\b'
    match = re.search(pattern, sentence.lower())
    print(f'Trigger "{trigger}" matches: {match is not None}')

