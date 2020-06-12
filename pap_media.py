from mediawiki import MediaWiki

wikipedia = MediaWiki(lang='pap',user_agent='code-for-nl-pap-parser')

bina_result = wikipedia.opensearch('bina', results=3)

print(bina_result)

