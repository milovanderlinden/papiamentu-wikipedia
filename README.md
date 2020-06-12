# Various tools for working with digital sources of the papiementu language

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build NLP files from a downloaded papiementu wikipedia file
```bash
./pap_create.sh
```

## Small test with the corpus
```bash
python test_pap_corpus.py
```

## Small test with wikipedia API 

```bash
python pap_media.py {search-term}
```

When you run this for instance with `bina` as term; the result will be:

```python
python pap_media.py bina
I found Biná which is a Animalnan
bina is known on wikidata with the code Q215887
Biná is called 'witstaarthert' in dutch
```
