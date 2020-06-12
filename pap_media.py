from mediawiki import MediaWiki
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api

wikipedia = MediaWiki(lang='pap',user_agent='code-for-nl-pap-parser')
wikidata = MediaWiki(url="https://www.wikidata.org/w/api.php",user_agent='code-for-nl-pap-parser')

bina_result = wikipedia.search('bina', results=1)

if len(bina_result) > 0:
    page = wikipedia.page(bina_result[0])
    print ("I found %s" % bina_result[0], "which is a","/".join(page.categories))
    #print(page.images)

    # Now I am going to search this one on wikidata, this will return a code. like Q215887
    bina_data = wikidata.search(bina_result[0], results=1)

    if len(bina_data) > 0:

        # First get the page. Read the images found
        data_page = wikidata.page(bina_result[0])
        print(data_page.images)

        # Now try the qwikidata interface
        Q_BINA = bina_data[0]
        entity = get_entity_dict_from_api(Q_BINA)
        q = WikidataItem(entity)
        print(q.get_label(lang="pap"), "is called", "'" + q.get_label(lang="nl") + "'", "in dutch")
        p18_claims = q.get_claim_group('P18')
