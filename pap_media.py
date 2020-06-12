import argparse
import urllib
from mediawiki import MediaWiki
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api


def main(search_term):
    wikipedia = MediaWiki(lang='pap', user_agent='code-for-nl-pap-parser')
    wikidata = MediaWiki(url='https://www.wikidata.org/w/api.php',
                         user_agent='code-for-nl-pap-parser')

    search_result = wikipedia.search(search_term, results=4)

    for result_item in search_result:
        page = wikipedia.page(result_item)
        print('I found page \'%s\' for term \'%s\'' % (result_item, search_term), 'with categories', '/'.join(
            page.categories), 'https://pap.wikipedia.org/wiki/' + urllib.parse.quote(result_item))
        # print(page.images)

        # Now I am going to search this one on wikidata, this will return a code. like Q215887
        search_data = wikidata.search(result_item, results=1)

        for data_item in search_data:
            Q_CODE = data_item
            print(result_item, 'is known on wikidata with the code',
                  Q_CODE, 'https://www.wikidata.org/wiki/' + Q_CODE)
            # Now try the qwikidata interface
            entity = get_entity_dict_from_api(Q_CODE)
            q = WikidataItem(entity)
            pap_data_label = q.get_label(lang='pap')
            nl_data_label  = q.get_label(lang='nl')
            if pap_data_label and nl_data_label:
                # First get the page. Read the images found
                data_page = wikidata.page(result_item)
                # print(data_page.images)

                print(pap_data_label, 'is called', nl_data_label, 'in dutch')
            elif pap_data_label and not nl_data_label:
                print(pap_data_label, 'has no entry for dutch!')
            elif not pap_data_label and nl_data_label:
                print(Q_CODE, 'does not match papiamentu entry')
            elif not pap_data_label and not nl_data_label:
                print(pap_data_label, 'has no entry for dutch or papiamentu!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter a search term')
    parser.add_argument('name', help='search term')
    args = parser.parse_args()

    main(args.name)
