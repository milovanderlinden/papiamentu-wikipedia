import argparse
from mediawiki import MediaWiki
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api


def main(search_term):
    wikipedia = MediaWiki(lang='pap',user_agent='code-for-nl-pap-parser')
    wikidata = MediaWiki(url="https://www.wikidata.org/w/api.php",user_agent='code-for-nl-pap-parser')

    search_result = wikipedia.search(search_term, results=1)

    if len(search_result) > 0:
        page = wikipedia.page(search_result[0])
        print ("I found %s" % search_result[0], "which is a","/".join(page.categories))
        #print(page.images)

        # Now I am going to search this one on wikidata, this will return a code. like Q215887
        search_data = wikidata.search(search_result[0], results=1)

        if len(search_data) > 0:
            Q_CODE = search_data[0]
            print(search_term, "is known on wikidata with the code", Q_CODE)

            # Now try the qwikidata interface
            entity = get_entity_dict_from_api(Q_CODE)
            q = WikidataItem(entity)
            pap_data_label = q.get_label(lang="pap")
            if pap_data_label:
                # First get the page. Read the images found
                data_page = wikidata.page(search_result[0])
                # print(data_page.images)

                print(q.get_label(lang="pap"), "is called", "'" + q.get_label(lang="nl") + "'", "in dutch")
                p18_claims = q.get_claim_group('P18')
            else:
                print(Q_CODE, "has no entry for papiamentu!")

        else:
            print(search_term, "could not be found on pap.wikipedia.org")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Enter a search term')
    parser.add_argument('name', help='search term')
    args = parser.parse_args()

    main(args.name)