#!/usr/bin/env python3

'''
Ways to tease out ontology annotations from textual metadata.
Includes exact match search and a search based on the term name
and its overlap with metadata.

Outputs a matrix of samples as rows and ontology term IDs as columns
(boolean or an arbitrary number indicating string similarity).

'''

#TODO onto.ancestry_table
#How to programatically extract words that do not contribute to good poredictions, such as 'control', 'feature'?
#proceed: onto.annotation_matrix
#TODO remove obsolete later
#TODO lower() the term names?

def foo():
    s2w = pd.DataFrame(s2w.toarray(), index = samples, columns = vocabulary)

#    #hits = {z:x[z].sort_values()[-3:].index.tolist() for z in x}
#    hits = {sid:z.sort_values()[-1:].index.tolist() for sid,z in res.items()}
#    hits = pd.DataFrame(hits).T
#    # get top 3
#    hits.columns = ['envo']
#    hits = hits.merge(pd.DataFrame(pd.Series(o.name_map())), left_on='envo', right_index=True)
#    hits = hits.merge(md, right_index=True, left_index=True, how='inner')
#    hits.to_csv('tmp.tsv', sep='\t')
#from difflib import SequenceMatcher
#SequenceMatcher(None,).ratio()

## this works but may be outdated. (IS, 2013)
#obo_url = 'https://raw.githubusercontent.com/xapple/seqenv/master/seqenv/data_envo/envo.obo'
#obo_url = 'http://environmentontology.org/downloads'
#
## len=all
#obo_url = 'https://raw.githubusercontent.com/EnvironmentOntology/envo/master/envo.obo'
#
## len=38
#obo_url = 'https://raw.githubusercontent.com/EnvironmentOntology/envo/master/subsets/EnvO-Lite-GSC.obo'
#
#g = obonet.read_url('/P/md2stuff/obo_data/envo-basic.obo') #len=2000
#id_to_name = {id_: data['name'] for id_, data in graph.nodes(data=True)}
#
#eo = pd.read_csv('obo_data/ENVO.csv')
## keep only informative
#eo = eo.ix[:, eo.apply(pd.isnull).sum() < eo.shape[0]*0.5]
#eo = eo.ix[ [not i for i in eo['Obsolete']], :]


import wrenlab_ontology_mod as wo
import re
from fuzzywuzzy import fuzz
import fuzzywuzzy as fw
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Memory

memory = Memory(cachedir="cache/")


@memory.cache
def process_md(md_fp):
    usecols = [
        'sample_id', 'sample_name',
        'centre_name', 'experimental_factor',
        'study_linkout', 'study_abstract', 'biome',
        #'keywords', 'group_word_count'
        ]
    sample_md = pd.read_csv(md_fp, sep='\t', usecols = usecols, index_col=False)
    sample_md.drop_duplicates(inplace=True)
    sample_md.reset_index(inplace=True)
    sample_md.index = sample_md['sample_id']

    sample_md.drop(['sample_id', 'index'], 1, inplace=True, errors='ignore')
    sample_md.drop_duplicates(inplace=True)

    #sample_md = {k:pool_md_(v) for k,v in sample_md.items()}
    return sample_md


def pool_md_(md_row):
    '''
    md_row: list or pd.Series
    '''
    all_words = []
    to_strip = ['"', "'", ";", " ", "?", ".", ",", "(", ")"]
    # TODO add condition to strip ':' if not matching 'BTO:'
    to_exclude = ['null', ' ', '-', 'nan', 'true', 'false', 'none']

    o = " ".join(md_row).split(" ")
    o = [i.lower() for i in o]
    for s in to_strip:
        o = [i.strip(s) for i in o]

    o = set([i for i in o if i not in to_exclude])
    return o


def get_stop_words():
    stop_words = list(ENGLISH_STOP_WORDS) +\
                ['04', '13', '36', 'gs', '03', '10', '1b', 'ph', '92', '12', '05', 'mt',
                'br', 'fr', '11', '25', '91', 'pm', 'cf', 'sp', '16', '19', '87', 'km',
                '01', '20', 'v1', 'uc', '15', 'o2', '23', '21', '14', '18', 'cd'] +\
                ['null', ' ', '-', 'nan', 'true', 'false', 'none']
                #['feature', 'environment'] # from md - somewhat cheaty
    remove_from_stop_words = ['bottom', 'side', 'fire', 'back', 'behind', 'well',
                'thin', 'empty', 'down'] # not exhaustive
    stop_words = [w for w in stop_words if w not in remove_from_stop_words]
    return stop_words


@memory.cache
def sid2word(sample_md, reduce=False, ontology=None):
    # This is manual. Does CountVectorizer accept conditional removal?
    # Like stop_words include len(word) < 3?
    md = {
            k:" ".join([str(i) for i in v.values()])\
            #k:[str(i) for i in v.values()]\
            #k:[str(i) for i in pd.Series(v)]\
                    for k,v in sample_md.T.to_dict().items()
            }

    #stop_words = get_stop_words()
    #cv = CountVectorizer(stop_words=stop_words) # reduce the search space

    cv = CountVectorizer(stop_words=get_stop_words())
    #TODO remove those features that are present for more than x % of documents
    s2w = cv.fit_transform([v for v in md.values()]) != 0
    samples = sample_md.index
    if reduce:
        all_vocabulary = pd.Index(cv.get_feature_names())
        vocabulary = reduce_vocabulary(all_vocabulary, ontology=ontology)
        s2w = s2w[:, [all_vocabulary.get_loc(i) for i in vocabulary]]
    else:
        vocabulary = pd.Index(cv.get_feature_names())

    s2w = pd.SparseDataFrame(s2w.toarray(), index = samples, columns = vocabulary)

    return s2w, vocabulary, samples

def get_onto_vocabulary(ontology, stop_words=None):
    o_voc = " ".join(ontology.terms.Name).split(' ')

    #stop_words = get_stop_words()
    if stop_words:
        o_voc = [w for w in o_voc if w not in stop_words]

    return o_voc

def reduce_vocabulary(vocabulary, ontology, mode='exact', stop_words=get_stop_words()):
    '''
    Reduce the vocabulary to only those words that match words
    in ontology names.

    mode:   'exact' exact matches
            float   with fuzzywuzzy.StringMatcher.ratio > mode
    '''
    onto_voc = get_onto_vocabulary(ontology, stop_words = stop_words)

    if mode == 'exact':
        return [w for w in vocabulary if w in onto_voc]
    elif isinstance(mode, float):
        return [w for w in vocabulary if
                max(
                    #match(w, onto_voc)
                    [fw.StringMatcher.ratio(w, o_w) for o_w in onto_voc]
                    ) > 0.85
                ]


### priority 1. just check if obo_id matches
def extract_ontology_annotation(metadata, ontology_prefix='ENVO'):
    '''
    ENA md, Envo:
        - 20 different studies have EnvO annotation
            > sample_md[['project_id']].align(envo, axis=0, join='inner')[0].drop_duplicates()
        - 26 different sets of annotations described
    '''
    md_path = '/D/ebi/metadata_ENA.tsv'

    envo_cols = ['biome', 'feature', 'material', 'env_biome', 'env_feature',
        'env_material', 'env_matter', 'environment (biome)',
        'environment (feature)', 'environment (material)',
        'Environment (featurel)', 'Environment (material)']

    md = pd.read_csv(md_path, sep='\t', low_memory=False, index_col=0, usecols=envo_cols)

    o = md[envo_cols].applymap(
                lambda s: re.search(ontology_prefix+':\d{4,12}', s) if isinstance(s,str) else None
            )\
            .dropna((0,1), 'all')

    envo = o.applymap(lambda s: s.group() if s else None)
    envo = {k:[
            v_ for v_ in v.values() if v_
        ]
        for k,v in envo.T.to_dict().items()}


### priority 2. Exact matches of ontology names
# (the longer - and less likely to match, hence more precise - will have a natural advantage)

### priority 3. matching: iterate over {sample_id:pd.Series(md for tha sample)}
def match_name_fuzzy(sample_md, terms):
    return pd.Series([ fuzz.partial_ratio(" ".join(sample_md), n) \
                        for n in terms['Name'] ],
                index = terms.index
                )

### match
@memory.cache
def md2onto(sample_md, ontology, mode, rm_obsolete=True):
    '''
    sample_md:  dictionary {sample_id:
                pd.Series or list representing the row of metadata}
    ontology:   an Ontology object
    mode:       Either 'id' for exact ID matching
                or 'name' for fuzzy name matching
    '''
    if rm_obsolete:
        terms = ontology.terms[ontology.terms.Name.str.find('obsolete') == -1]
        #{k:v for k,v in onto if v.find('obsolete') == -1}

    o = {}
    if mode == 'id':
        for sid, smd in sample_md.items():
            o[sid] = match_id_exact(smd, terms)
    elif mode == 'name':
        for sid, smd in sample_md.items():
            o[sid] = match_name_fuzzy(smd, terms)
    #o = pd.DataFrame.from_dict(o)
    return o



def match(term_name, word_list, mode='exact'):
    '''
    Takes in a ontology term name / description and a vector of words
    and returns a measure of string similarity for each word [vector]

    mode:  'exact', ...
    '''
    if mode == 'partial_ratio':
        return pd.Series([ fuzz.partial_ratio(term_name, w) \
                            for w in word_list ],
                    index = word_list
                    )
    if mode == 'token_sort_ratio':
        return pd.Series([ fuzz.token_sort_ratio(term_name, w) \
                            for w in word_list ],
                    index = word_list
                    )
    if mode == 'ratio':
        return pd.Series([ fuzz.ratio(term_name, w) \
                            for w in word_list ],
                    index = word_list
                    )
    if mode == 'sequence_matcher':
        return pd.Series([ fuzz.SequenceMatcher(term_name, w) \
                            for w in word_list ],
                    index = word_list
                    )
    if mode == 'exact':
        return pd.Series([ sum([t == w for t in term_name.split(' ')])\
                        for w in word_list],
                    index = word_list
                    )
    if mode == 'exact_weighted':
        return pd.Series([ sum([t == w for t in term_name.split(' ')]) / len(term_name.split(' '))\
                        for w in word_list],
                    index = word_list
                    )
    else:
        print('Specify mode')


@memory.cache
def word2term(ontology, vocabulary, mode):
    '''
    Generate word scores for each of the ontology terms
    '''
    w2t = {}
    for t,n in zip(ontology.terms.index, ontology.terms.Name):
    # match onto name with word in vocabulary
        w2t[t] = match(n, vocabulary, mode)
    w2t = pd.DataFrame(w2t)
    return w2t


def sid2term(samples, vocabulary, s2w, w2t, mode='exact'):
    # USES s2w as DataFrame
    #TODO OPTIM use a sparse matrix format instead
    #s2w = pd.DataFrame(s2w.toarray(), index=samples, columns=vocabulary)
    #s2w = csr_matrix(s2w)
    s2t = {}
    for sid in samples:
        print('--- --- ', sid)
        words = s2w.T[sid]
        subM = w2t[words]
        scores = subM.sum(0)

        ## CHOICE
        #s2t_ = s2t_.index[s2t_ == s2t_.max()]
        #s2t_ = s2t_[s2t_ == s2t_.max()]
        #scores = scores[scores > np.percentile(scores, 80)]
        scores = scores[scores > scores.max() * .8].sort_values(ascending=False)#[:5]

        s2t[sid] = scores

    return s2t


def s2t_table(s2t, md, onto):
    df = { k: {
                    i:{'term_id':i, 'term_name': onto.terms.Name[i], 'score':v[i],
                    'sample_id': k, 'biome': md.biome[k], 'study_linkout': md.study_linkout[k]
                } for i in v.index
            } for k,v in s2t.items() }

    df = {k:pd.DataFrame(v) for k,v in df.items()}


def s2t_table_even(s2t, md, onto):
    df = {k:v.index[0] for k,v in s2t.items() if len(v) >0}
    df = pd.DataFrame(pd.Series(df), columns = ['term_id'])
    df = md[['biome', 'study_linkout']].merge(df, left_index=True, right_index=True).reset_index()
    term_names = onto.terms.reset_index()[['Name', 'TermID']]
    term_names.rename(columns = {'TermID':'term_id', 'Name': 'term_name'}, inplace=True)
    df = df.merge(term_names)
    df.rename(columns={'index':'sample_id'}, inplace=True)
    df.index = df['sample_id']
    #df.drop('index', 1, inplace=True)
    df = df[['term_id', 'sample_id', 'term_name', 'biome', 'study_linkout']]
    return df



def main():
    print('--- Loading data...')
    ofile = 'md2onto.pkl'
    obo_location = '/P/md2stuff/obo_data/envo-basic.obo'
    md_fp = '/D/ebi/DEFAULT_METADATA.tsv'

    onto = wo.fetch(obo_location, int_keys=False) # has obsolete

    print('--- Drawing world map...')
    md = process_md(md_fp)
    md = md.sample(200, random_state=1)
    s2w, vocabulary, samples = sid2word(md, reduce=True, ontology = onto)

    print('--- Scoring words...')
    w2t = word2term(onto, vocabulary, mode = 'exact_weighted')
    #w2t = word2term(onto, vocabulary, mode = 'ratio')
    #w2t = word2term(onto, vocabulary, mode = 'sequence_matcher')
    #w2t = word2term(onto, vocabulary, mode = 'token_sort_ratio')
    print('--- Collecting meteor particles...')
    #print('--- Annotating...')

    s2t = sid2term(samples, vocabulary, s2w, w2t)

    df = s2t_table_even(w2t, md, onto)
    df.to_csv('st2.tsv', sep='\t')

    o = {}
    o['s2w'] = s2w
    o['w2t'] = w2t
    o['s2t'] = s2t
    o['samples'] = samples
    o['vocabulary'] = vocabulary
    pickle.dump(o, open(ofile, 'wb'))
    print('--- Done.')


def get_synonyms(ontology)
    '''
    Get a matrix of association strength between each of the terms.
    E.g. 1 is identity link ('is_a' relationhip); 0.5 is all other relationships.
    Query columns for relationships.
    '''
    o = {k:
                {k_:v_['type'] for k_,v_ in v.items()}
            for k,v in ontology.graph.edge.items()
            if len(v) > 1}
    # For now: just make ones for any relationship
    o = pd.DataFrame(o).notnull()
    return o


if __name__ == '__main__':
    main()


