source venv/bin/activate

curl -fSL -R -J -O 'https://dumps.wikimedia.org/papwiki/latest/papwiki-latest-pages-articles.xml.bz2'

python -m gensim.scripts.make_wiki ./papwiki-latest-pages-articles.xml.bz2 corpus/wiki_pap