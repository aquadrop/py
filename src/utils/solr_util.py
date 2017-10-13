from SolrClient import SolrClient

SOLR_URL = 'http://localhost:11403/solr'

solr_client = SolrClient(SOLR_URL)


def solr_qa(core, query):
    params = {'q': query, 'q.op': 'or'}
    responses = solr_client.query(core, params)
    docs = responses.docs
    return docs
