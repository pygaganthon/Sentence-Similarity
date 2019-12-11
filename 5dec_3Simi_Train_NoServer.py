#Gagandeep Singh Jossan gxj170003
#Maitreyee Mhasakar mam171630

import pandas as pd
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
from nltk.corpus.reader import NOUN
from nltk.corpus.reader import VERB
#Load dataset into dataframe
#df=pd.read_csv('C:/Users/gagan/OneDrive/Desktop/NLPProject/data/sample_train.txt', sep="\t",error_bad_lines=False)
df=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//data/train-set.txt', sep="\t",error_bad_lines=False)

df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

#print(df)
#Prepare Dataframe and name columns
df.columns=["id","Sentence1","Sentence2","Gold Tag"]
df_features=df.copy()

   
def normalize(df):
    result = df.copy()
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag, isOtherPOSRequired):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    elif isOtherPOSRequired==True:
        return 'n'
    return 't'


#*************Creating Dependency Parser for senetcnes in dataset**********
#import spacy
##Load the english module
#nlp = spacy.load("en")
#
#Dep_Sentence1=[]
#Dep_Sentence2=[]
#dependencies=['nsubj','dobj','iobj','ccomp','xcomp','nominal','nmod','amod','nummod','appos','det','case']
#postags=['VB','VBD','VBG','VBN','VBP','VBZ','NN','NNS','NNP','NNPS']
#for index, row in df_features.iterrows():
#    sen1= nlp(row['Sentence1'])
#    sen2= nlp(row['Sentence2'])
#    if index==0:
#        print(sen1)
#        print(sen2)
#    dep_sen1=[]
#    dep_sen2=[]
#    for token1 in sen1:
#        if token1.tag_ in postags or token1.head.tag_ in postags:
#            dep_sen1.append((token1.text,token1.dep_, token1.head.text))
#    for token2 in sen2:
#        if token2.tag_ in postags or token2.head.tag_ in postags:
#            dep_sen2.append((token2.text,token2.dep_, token2.head.text))
#    Dep_Sentence1.append(dep_sen1)
#    Dep_Sentence2.append(dep_sen2)

       
##Store dependency parsered sentences in dataframe
#df_features["DepPar_Sentence1"] = Dep_Sentence1
#df_features["DepPar_Sentence2"] = Dep_Sentence2


from nltk.tokenize import word_tokenize
df_features["Token_Sentence1"] = df["Sentence1"].apply(word_tokenize)
df_features["Token_Sentence2"] = df["Sentence2"].apply(word_tokenize)

common_token_list=[]
for index, row in df_features.iterrows():
    common_token=set(row['Token_Sentence1']) and set(row['Token_Sentence2'])
    common_token_list.append(common_token)

df_features["Common_tokens"] = common_token_list

postag_Sentence1=[]
postag_Sentence2=[]
for index, row in df_features.iterrows():
    sen1temp=[]
    sen2temp=[]
    for i in row['Token_Sentence1']:
        sen1temp.append(i)
    for j in row['Token_Sentence2']:
        sen2temp.append(j)
    postag_Sentence1.append(dict(pos_tag(sen1temp)))
    postag_Sentence2.append(dict(pos_tag(sen2temp)))
   
   
df_features["PosTag_Sentence1"] = postag_Sentence1
df_features["PosTag_Sentence2"] = postag_Sentence2  
   
lemmatizer = WordNetLemmatizer()
lemmatized_Sentence1=[]
lemmatized_Sentence2=[]
postaglemma_Sentence1=[]
postaglemma_Sentence2=[]
for index, row in df_features.iterrows():
    sen1temp=[]
    sen2temp=[]
    for idx,i in enumerate(row['Token_Sentence1']):
#        print(penn_to_wn(df_features['PosTag_Sentence1'][index][idx][1]))
        w1 =lemmatizer.lemmatize(i,penn_to_wn(df_features['PosTag_Sentence1'][index][i],True))
        sen1temp.append(w1)
    for idx,j in enumerate(row['Token_Sentence2']):
#        print(penn_to_wn(df_features['PosTag_Sentence2'][index][idx][1]))
        w2 =lemmatizer.lemmatize(j,penn_to_wn(df_features['PosTag_Sentence2'][index][j],True))
        sen2temp.append(w2)
    postaglemma_Sentence1.append(dict(pos_tag(sen1temp)))
    postaglemma_Sentence2.append(dict(pos_tag(sen2temp)))
    lemmatized_Sentence1.append(sen1temp)
    lemmatized_Sentence2.append(sen2temp)




df_features["Lema_Sentence1"] = lemmatized_Sentence1
df_features["Lema_Sentence2"] = lemmatized_Sentence2

df_features["PosTagLema_Sentence1"] = postaglemma_Sentence1
df_features["PosTagLema_Sentence2"] = postaglemma_Sentence2


#sen1ptags=[]
#sen2ptags=[]
#for index, row in df_features.iterrows():
#
#    ttagss1 = {}
#    doc1 = nlp(row['Sentence1'])
#    for token in doc1:
#        ttagss1[token.text] = token.tag_
#    sen1ptags.append(ttagss1)
#    
#    ttagss2 = {}
#    doc2 = nlp(row['Sentence2'])
#    # record all possible edges
#    for token in doc2:
#        ttagss2[token.text] = token.tag_
#    sen2ptags.append(ttagss2)
#  
#    
#df_features["Token_POS_Sent_1"] = sen1ptags
#df_features["Token_POS_Sent_2"] = sen2ptags

tknsen1ptags=[]
tknsen2ptags=[]
lema_token_s1 = []
lema_token_s2 = []
for index, row in df_features.iterrows():

    lema1 = []
    doc1 = nlp(row['Sentence1'])
    for token in doc1:
        lemw1 = lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True))
        lema1.append(lemw1)
       
    tknsen1ptags.append(dict(pos_tag(lema1)))
    lema_token_s1.append(lema1)
#    lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)),
    lema2 = []
    doc2 = nlp(row['Sentence2'])
    for token in doc2:
        lemw2 = lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True))
        lema2.append(lemw2)
       
    tknsen2ptags.append(dict(pos_tag(lema2)))
    lema_token_s2.append(lema2)
   
df_features["Token_Lema_Sent_1"] = lema_token_s1
df_features["Token_Lema_Sent_2"] = lema_token_s2
df_features["Token_LemaPOS_Sent_1"] = tknsen1ptags
df_features["Token_LemaPOS_Sent_2"] = tknsen2ptags



listOfHypernymSent1 = []
listOfHypernymSent2 = []

listOfHyponymSent1 = []
listOfHyponymSent2 = []

listOfHolonymSent1 = []
listOfHolonymSent2 = []

listOfMeronymSent1 = []
listOfMeronymSent2 = []

from nltk.corpus import wordnet
for index, row in df_features.iterrows():
    listOfSSHypePair = []
    listOfSSHypoPair = []
    listOfSSHoloPair = []
    listOfSSMeroPair = []
    for word in row['Token_Sentence1']:
      
        for ss in wordnet.synsets(word):
            listOfSSHypePair.extend(ss.hypernyms())
            listOfSSHypoPair.extend(ss.hyponyms())
            listOfSSHoloPair.extend(ss.part_holonyms())
            listOfSSMeroPair.extend(ss.part_meronyms())
               
           
       

    listOfHypernymSent1.append(set(listOfSSHypePair))  
    listOfHyponymSent1.append(set(listOfSSHypoPair))  
    listOfHolonymSent1.append(set(listOfSSHoloPair))  
    listOfMeronymSent1.append(set(listOfSSMeroPair))
   
    listOfSSHypePair = []
    listOfSSHypoPair = []
    listOfSSHoloPair = []
    listOfSSMeroPair = []
       

    for word in row['Token_Sentence2']:
        for ss in wordnet.synsets(word):
            listOfSSHypePair.extend(ss.hypernyms())
            listOfSSHypoPair.extend(ss.hyponyms())
            listOfSSHoloPair.extend(ss.part_holonyms())
            listOfSSMeroPair.extend(ss.part_meronyms())
               

    listOfHypernymSent2.append(set(listOfSSHypePair))  
    listOfHyponymSent2.append(set(listOfSSHypoPair))  
    listOfHolonymSent2.append(set(listOfSSHoloPair))
    listOfMeronymSent2.append(set(listOfSSMeroPair))  
    
    

   
df_features['Hyper_Sentence1'] = listOfHypernymSent1
df_features['Hyper_Sentence2'] = listOfHypernymSent2

df_features['Hypo_Sentence1'] = listOfHyponymSent1
df_features['Hypo_Sentence2'] = listOfHyponymSent2

df_features['Holo_Sentence1'] = listOfHolonymSent1
df_features['Holo_Sentence2'] = listOfHolonymSent2

df_features['Mero_Sentence1'] = listOfMeronymSent1
df_features['Mero_Sentence2'] = listOfMeronymSent2





## Counties with population declines will be Vermillion, Posey and Madison.
#[('Counties', 'nsubj', 'be'), ('with', 'prep', 'Counties'), ('population', 'compound', 'declines'), ('declines', 'pobj', 'with'),
# ('will', 'aux', 'be'), ('be', 'ROOT', 'be'), ('Vermillion', 'attr', 'be'), (',', 'punct', 'Vermillion'),
# ('Posey', 'conj', 'Vermillion'), ('and', 'cc', 'Posey'), ('Madison', 'conj', 'Posey'), ('.', 'punct', 'be')]
#
## Vermillion, Posey and Madison County populations will decline.
#[('Vermillion', 'nmod', 'populations'), (',', 'punct', 'Vermillion'), ('Posey', 'conj', 'Vermillion'),
# ('and', 'cc', 'Posey'), ('Madison', 'compound', 'County'), ('County', 'conj', 'Posey'), ('populations', 'nsubj', 'decline'),
# ('will', 'aux', 'decline'), ('decline', 'ROOT', 'decline'), ('.', 'punct', 'decline')]    



# =============================================================================
# # gives:
# {'a': {'conclusive'},
#  'n': {'conclusion', 'conclusions', 'conclusivenesses', 'conclusiveness'},
#  'r': {'conclusively'},
#  'v': {'concludes', 'concluded', 'concluding', 'conclude'}}
# =============================================================================

from nltk import pos_tag

postag_Sentence1=[]
postag_Sentence2=[]
#List to store nouns and verbs in common in both sentences/total number of nouns and verbs in the sentence repectively.
Common_nouns_sen12=[]
Common_verbs_sen12=[]

arr=[]
arr1=[]

from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
def  getSimilarity(sim, word1, word2, pos1, pos2):
    lst = []
    try:
        word1synsets = wn.synsets(word1, pos1)
    except:
        if(sim=="path_similarity"):
            val_sim = 0.1
        if(sim=="lch_similarity"):
            val_sim = 1.2
        if(sim=="wup_similarity"):
            val_sim = 0.1
#        if(sim=="res_similarity"):
#            val_sim = 0.1
#        if(sim=="jcn_similarity"):
#            val_sim = 0.1
#        else:
#            val_sim = 0.1
        lst.append(val_sim)
        return pd.Series(lst).dropna()
    try:
        word2synsets = wn.synsets(word2, pos2)
    except:
        lst.append(0.1);
        return pd.Series(lst).dropna()
   
    for w1 in word1synsets:
        for w2 in word2synsets:
            try:
                if(sim=="path_similarity"):
                    val_sim = w1.path_similarity(w2)
                if(sim=="lch_similarity"):
                    val_sim = w1.lch_similarity(w2)
                if(sim=="wup_similarity"):
                    val_sim = w1.wup_similarity(w2)
#                if(sim=="res_similarity"):
#                    val_sim = w1.res_similarity(w2, brown_ic)
#                    if(val_sim > 3.9):
#                        val_sim = 3.9
#                if(sim=="jcn_similarity"):
#                    ic1, ic2, lcs_ic = wn._lcs_ic(w1, w2, brown_ic)
#                    icsum = (ic1 + ic2)
#                    val_sim = w1.jcn_similarity(w2,  brown_ic)
#                    if(val_sim > 10 and ic1 == ic2 and ic2 == lcs_ic):
#                        val_sim = 0.1
#                    if(val_sim > 10 and (icsum == 2*lcs_ic)):
#                        val_sim = 10
#                if(sim=="lin_similarity"):
#                    val_sim = w1.lin_similarity(w2, semcor_ic)
            except:
                val_sim = -1
            if(val_sim == None or val_sim == -1 or val_sim < 0.01):
                if(sim=="path_similarity"):
                    val_sim = 0.1
                if(sim=="lch_similarity"):
                    val_sim = 1.2
                if(sim=="wup_similarity"):
                    val_sim = 0.1
#                if(sim=="res_similarity"):
#                    val_sim = 0.1
#                if(sim=="jcn_similarity"):
#                    val_sim = 0.1
#                else:
#                    val_sim = 0.1
            lst.append(val_sim)
    if(len(lst)==0):
        if(sim=="path_similarity"):
            val_sim = 0.1
        if(sim=="lch_similarity"):
            val_sim = 1.2
        if(sim=="wup_similarity"):
            val_sim = 0.1
#        if(sim=="res_similarity"):
#            val_sim = 0.1
#        if(sim=="jcn_similarity"):
#            val_sim = 0.1
#        else:
#            val_sim = 0.1
        lst.append(val_sim)
    return pd.Series(lst).dropna()

HyperCommonDone = False
def getNormalizedCount(sim):
    Common_nouns_sen12=[]
    Common_verbs_sen12=[]
    comon_hyper_score = []
    for index, row in df_features.iterrows():
        list_of_nouns_sen1=[]
        list_of_verbs_sen1=[]
     
        list_of_nouns_sen2=[]
        list_of_verbs_sen2=[]
 
        #Variables to store count of number of nouns and verbs in the sentence.
        common_nouns=0
        common_verbs=0
    #Generate list of Nouns and Verbs in Sentence 1 and 2.
     
        for i in pos_tag(row['Lema_Sentence1']):
    #        if i[1] in Nouns:
            if penn_to_wn(i[1],False)=='n':
                #Actual lematized token is appended to the list
                list_of_nouns_sen1.append(i[0])
            elif penn_to_wn(i[1],False)=='v':
                list_of_verbs_sen1.append(i[0])
 
        for j in pos_tag(row['Lema_Sentence2']):
    #        if j[1] in Nouns:
            if penn_to_wn(j[1],False)=='n':
                list_of_nouns_sen2.append(j[0])
 
    #        elif j[1] in Verbs:
            elif penn_to_wn(j[1],False)=='v':
                list_of_verbs_sen2.append(j[0])
 
 
        #Compare nouns from sentence 1 and 2
        if HyperCommonDone != True:
            score_common_hyp = 0
            lstcm_hypernyms = []
        for word in list_of_nouns_sen1:
            for wordsen2 in list_of_nouns_sen2:
               
                if HyperCommonDone != True:
                    try:
                        wordsyns = wn.synset(str(word) + '.n.01')
                        wordsens2syns = wn.synset(str(wordsen2) + '.n.01')
                        common_hyper_2words = wordsyns.lowest_common_hypernyms(wordsens2syns)
                        lstcm_hypernyms.extend(common_hyper_2words)
                        for hyp in common_hyper_2words:
                            score_common_hyp = score_common_hyp + hyp.min_depth()
                    except:
                        score_common_hyp = score_common_hyp
                       
                if word==wordsen2:
                    common_nouns+=1
                elif (word in wordsen2 or wordsen2 in word):
                    common_nouns+=1              
                else:
                    h=0
                    if(len(wn.synsets(word,pos='n'))==0 or len(wn.synsets(wordsen2,pos='n'))==0):
                        h=0.1
                    else:
                        tmp = getSimilarity(sim, word, wordsen2, NOUN, NOUN)
                        if len(tmp)>0:
                            h = tmp.max()
                    if h>0.7:
                        common_nouns+=1
        #Compare verbs from sentence 1 and 2
        for word1 in list_of_verbs_sen1:
            for word1sen2 in list_of_verbs_sen2:
               
                if HyperCommonDone != True:
                    try:
                        wordsyns = wn.synset(str(word1) + '.v.01')
                        wordsens2syns = wn.synset(str(word1sen2) + '.v.01')
                        common_hyper_2words = wordsyns.lowest_common_hypernyms(wordsens2syns)
                        lstcm_hypernyms.extend(common_hyper_2words)
                        for hyp in common_hyper_2words:
                            score_common_hyp = score_common_hyp + hyp.min_depth()
                    except:
                        score_common_hyp = score_common_hyp
                       
                if word1==word1sen2:
                    common_verbs+=1        
                elif (word1 in word1sen2 or word1sen2 in word1):
                    common_verbs+=1              
                else:
                    h=0
                    if(len(wn.synsets(word1,pos='v'))==0 or len(wn.synsets(word1sen2,pos='v'))==0):
                        h=0.1
                    else:
                        tmp = getSimilarity(sim, word1, word1sen2, VERB, VERB)
                        if len(tmp)>0:
                            h = tmp.max()
                    if h>0.2:
                        common_verbs+=1
 
        if HyperCommonDone != True:              
            if(score_common_hyp==0):
                comon_hyper_score.append(0)
            else:
                comon_hyper_score.append(score_common_hyp / len(set(lstcm_hypernyms)))
        total_nn=len(list_of_nouns_sen1)*len(list_of_nouns_sen2)
        total_vb=len(list_of_verbs_sen1)*len(list_of_verbs_sen2)
 
        if len(list_of_nouns_sen1)==0 and len(list_of_nouns_sen2)==0:
            Common_nouns_sen12.append(0)
        elif len(list_of_nouns_sen1)==0 or len(list_of_nouns_sen2)==0:
            Common_nouns_sen12.append(0)
         
        else:
            Common_nouns_sen12.append(((common_nouns*100)/total_nn))
         
        if len(list_of_verbs_sen1)==0 and len(list_of_verbs_sen2)==0:
            Common_verbs_sen12.append(0)
        elif len(list_of_verbs_sen1)==0 or len(list_of_verbs_sen2)==0:
            Common_verbs_sen12.append(0)
         
        else:
            Common_verbs_sen12.append(((common_verbs*100)/ total_vb))


    Common_nouns_sen12_new = []
    sumn = sum(Common_nouns_sen12)
    if(sumn!=0):
        for i in Common_nouns_sen12:
            Common_nouns_sen12_new.append((i*100)/sumn)
    else:
        Common_nouns_sen12_new = Common_nouns_sen12
     
    Common_verbs_sen12_new = []
    sumv = sum(Common_verbs_sen12)
    if(sumv!=0):
        for i in Common_verbs_sen12:
            Common_verbs_sen12_new.append((i*100)/sumv)
    else:
        Common_verbs_sen12_new = Common_verbs_sen12
    return Common_nouns_sen12_new, Common_verbs_sen12_new, comon_hyper_score
 

similarities = ["path_similarity","lch_similarity", "wup_similarity"]
#                , "res_similarity", "jcn_similarity", "lin_similarity"]

for sim in similarities:
    Common_nouns_sen12_new, Common_verbs_sen12_new, comon_hyper_score = getNormalizedCount(sim)
    df_features["Norm_NN_by" + sim[0:3]] = Common_nouns_sen12_new
    df_features["Norm_VB_by" + sim[0:3]] = Common_verbs_sen12_new
    if HyperCommonDone != True:
        df_features["Common_Hyper_Score"] = comon_hyper_score
        HyperCommonDone = True





prep={'of':1,'in':2,'to':3,'for':4,'with':5,'on':6,'at':7,'from':8,'by':9,'about':10,'as':11,'into':12,'like':13,'through':14,'after':15,'over':16,'between':17,'out':18,'against':19,'during':20,'without':21,'before':22,'under':23,'around':24,'among':25}
def Prep(df_features):
    List_common_prep=[]
    for index, row in df_features.iterrows():
        sen1_prep=[]
        sen2_prep=[]

        for i in row['Token_Sentence1']:
            if i in prep:
                sen1_prep.append(i)

        for j in row['Token_Sentence2']:
            if j in prep:
                sen2_prep.append(j)
        Common_prep=[]

        for k in sen1_prep:
            if k in sen2_prep:
                Common_prep.append(k)
        List_common_prep.append((len(Common_prep)/25)*100)
    return List_common_prep
   
   

               

df_features['Common_prepositions']= Prep(df_features)






print("Before Dependency Parser operations")
#nlp = StanfordCoreNLP('http://localhost:9000')
#
#from nltk.parse.stanford import StanfordDependencyParser
##path_to_jar = 'C://Users//gagan//OneDrive//Desktop//NLPProject//stanford-parser-full-2018-10-17//stanford-parser.jar'
##path_to_models_jar = 'C://Users//gagan//OneDrive//Desktop//NLPProject//stanford-parser-full-2018-10-17//stanford-parser-3.9.2-models.jar'
#path_to_jar = 'C://Users//maitr//OneDrive//Desktop//Project_Final//stanford-parser-full-2018-10-17//stanford-parser.jar'
#path_to_models_jar = 'C://Users//maitr//OneDrive//Desktop//Project_Final//stanford-parser-full-2018-10-17//stanford-parser-3.9.2-models.jar'
#

#dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)


lemmatizer = WordNetLemmatizer()
verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
noun = ['NN','NNS','NNP','NNPS']
nounverb = ['VB','VBD','VBG','VBN','VBP','VBZ','NN','NNS','NNP','NNPS']
othertags = ['LS', 'TO',  'WP', 'UH', 'JJ', '--',  'DT', 'PRP', ':',
           'WP$','PRP$', 'WDT', '(', ')', '.', ',', '``', '$',
           'RB', 'RBR', 'RBS',  'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'WRB',
           'EX', 'SYM', 'CC', 'CD', 'POS']
def returnCategory(tag):
    if(tag in noun):
        return NOUN
    if(tag in verb):
        return VERB
    else:
        return NOUN


import networkx as nx

 
   

nlp = spacy.load('en')
       
def getDependencyPairSim(sim):
    edgePathDepWtSum = []
    alledgePathDepWtSum = []
    tagDepWtSum = []
    setTagDepWtSum = []
    setOvefitEdgePathDepWtSum = []
    for index,row in df_features.iterrows():
   
        tagrelations1 = []
        wordrelations1 = []
        allwordrelations1 = []
       
        doc1 = nlp(row['Sentence1'])
        root1=[]
       
        # record all possible edges
        for token in doc1:
            tagrelations1.append((token.head.tag_, token.tag_))
            if(token.head.text == token.text):
                root1.append((lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True)), token.tag_))
            else:
                allwordrelations1.append(((lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)),token.head.tag_), (lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True)),token.tag_)))
            if(nounverb.__contains__(token.tag_) and nounverb.__contains__(token.head.tag_)):
#                print(lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)),token.head.tag_, lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True)), token.tag_)
                wordrelations1.append((lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)), lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True))))
                       

       
        G = nx.DiGraph()
        G.add_edges_from(allwordrelations1)
   
        paths1=[]
        for node in G:
            if G.out_degree(node)==0: #it's a leaf
                for root in root1:
                    try:
                        paths1.append(nx.shortest_path(G, root, node))
                        break
                    except:
                        continue
   
           
        allpathrelations1=[]
        #get all possible one edge paths invololving N and V and store it
        for eachpath in paths1:
            for i in range(len(eachpath)):
                for j in range(i+1,len(eachpath)):
                    if(nounverb.__contains__(eachpath[i][1]) and nounverb.__contains__(eachpath[j][1])):
                        allpathrelations1.append((eachpath[i][0],eachpath[j][0]))
   
    #    print(set(allpathrelations1))
        tagrelations2 = []
        wordrelations2 = []
        allwordrelations2 = []
   
        pathSimWt = []
       
        root2 = []
        doc2 = nlp(row['Sentence2'])
         # record all possible edges
        for token in doc2:
            if(token.head.text == token.text):
                root2.append((lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True)), token.tag_))
            else:
                allwordrelations2.append(((lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)),token.head.tag_), (lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True)),token.tag_)))
            tagrelations2.append((token.head.tag_, token.tag_))
            if(nounverb.__contains__(token.tag_) and nounverb.__contains__(token.head.tag_)):
                wordrelations2.append((lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)), lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True))))
   
    #    print(tagrelations2)
        G = nx.DiGraph()
        G.add_edges_from(allwordrelations2)
        paths2=[]
        # get all root to tail paths
        for node in G:
            if G.out_degree(node)==0: #it's a leaf
                for root in root2:
                    try:
                        paths2.append(nx.shortest_path(G, root, node))
                        break
                    except:
                        continue
    #    print(paths2)
        #get all possible one edge paths invololving N and V and store it
        allpathrelations2=[]
        for eachpath in paths2:
            for i in range(len(eachpath)):
                for j in range(i+1,len(eachpath)):
                    if(nounverb.__contains__(eachpath[i][1]) and nounverb.__contains__(eachpath[j][1])):
                        allpathrelations2.append((eachpath[i][0],eachpath[j][0]))
       
    #    print(set(allpathrelations2))        
        if len(wordrelations2+wordrelations1)!=0:
            setOvefitEdgePathDepWtSum.append(len(tuple(set(wordrelations2).intersection(set(wordrelations1)))) / len(set(wordrelations2+wordrelations1)))
        else:
            setOvefitEdgePathDepWtSum.append(0.8)
        setTagDepWtSum.append(len(tuple(set(tagrelations1).intersection(set(tagrelations2)))) / len(set(tagrelations2+tagrelations1)))
   
       
        #pass tags also
        #smaller on outer
        new_wordrelations1 = wordrelations1
        new_wordrelations2 = wordrelations2
        swapped=False
        if(len(set(wordrelations1)) > len(set(wordrelations2))):
            swapped=True
            new_wordrelations1 = wordrelations2
            new_wordrelations2 = wordrelations1
        pathSimWt=0
        for wr1 in set(new_wordrelations1):    
            maxh=0;
            maxd=0;
            for wr2 in new_wordrelations2:
                h=0
                d=0
                if(wr1[0]==wr2[0]):
                    maxh = 1
                else:
                    if(swapped):
#                        print(wr1[0], wr2[0] , df_features["Token_LemaPOS_Sent_2"][index][wr1[0]],df_features["Token_LemaPOS_Sent_1"][index][wr2[0]])
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr1[0]]), returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr2[0]]))
                    else:
#                        print(wr2[0], wr1[0] , df_features["Token_LemaPOS_Sent_2"][index][wr2[0]],df_features["Token_LemaPOS_Sent_1"][index][wr1[0]])
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr1[0]]), returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr2[0]]))
                    if len(tmp)>0:
                        h = tmp.max()
                    else:
                        h = 0.1
                if(wr1[1]==wr2[1]):
                    maxd = 1
                else:
                    if(swapped):
#                        print(wr1[1], df_features["PosTagLema_Sentence2"][index][wr1[1]], wr2[1], df_features["PosTagLema_Sentence1"][index][wr2[1]])
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr1[1]]), returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr2[1]]))
                    else:
#                        print(wr1[1], df_features["PosTagLema_Sentence1"][index][wr1[1]], wr2[1], df_features["PosTagLema_Sentence2"][index][wr2[1]])
#                        print(wr1[1] , wr2[1])
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr1[1]]), returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr2[1]]))
                    if len(tmp)>0:
                        d = tmp.max()
                    else:
                        d = 0.1
                if(h and maxh<h):
                    maxh=h
                if(d and maxd<d):
                    maxd=d
                if(maxh==1 and maxd==1):
                    break
            # to be modified
            if(maxh>0.2 and maxd>0.1):
                pathSimWt = pathSimWt + (maxh+maxd)/2;
            else:
                pathSimWt = 0.1 + pathSimWt;
        if len(new_wordrelations1)!=0:  
            edgePathDepWtSum.append(pathSimWt/len(set(new_wordrelations1)))
        else:
            edgePathDepWtSum.append(0.5)
       
       
        new_allpathrelations1 = allpathrelations1
        new_allpathrelations2 = allpathrelations2
       
        swapped = False
        if(len(set(allpathrelations1)) > len(set(allpathrelations2))):
            swapped = True
            new_allpathrelations1 = allpathrelations2
            new_allpathrelations2 = allpathrelations1
        pathSimWt=0
        for wr1 in set(new_allpathrelations1):  
            maxh=0;
            maxd=0;
            for wr2 in set(new_allpathrelations2):
                h=0
                d=0
                if(wr1[0]==wr2[0]):
                    maxh = 1
                else:
                    if(swapped):
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr1[0]]), returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr2[0]]))
                    else:
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr1[0]]), returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr2[0]]))
                    if len(tmp)>0:
                        h = tmp.max()
                    else:
                        h = 0
                if(wr1[1]==wr2[1]):
                    maxd = 1
                else:
                    if(swapped):
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr1[1]]), returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr2[1]]))
                    else:
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(df_features["Token_LemaPOS_Sent_1"][index][wr1[1]]), returnCategory(df_features["Token_LemaPOS_Sent_2"][index][wr2[1]]))
                    if len(tmp)>0:
                        d = tmp.max()
                    else:
                        d = 0
                if(h and maxh<h):
                    maxh=h
                if(d and maxd<d):
                    maxd=d
                if(maxh==1 and maxd==1):
                    break
            pathSimWt = pathSimWt + (maxh+maxd)/2;
        if len(new_allpathrelations1)!=0:
            alledgePathDepWtSum.append(pathSimWt/len(set(new_allpathrelations1)))
        else:
            alledgePathDepWtSum.append(0.5)
       
    #    print(j)
        #! smaller on outside
        t=0
        new_tagrelations1 = tagrelations1
        new_tagrelations2 = tagrelations2
       
        if(len(tagrelations1) > len(tagrelations2)):
            new_tagrelations1 = tagrelations2
            new_tagrelations2 = tagrelations1
        for tr1 in set(new_tagrelations1):
            for tr2 in set(new_tagrelations2):
                if(tr1[0]==tr2[0] and tr1[1]==tr2[1]):
                    t=t+1
                    break
        tagDepWtSum.append(t/(len(new_tagrelations1)));
    return alledgePathDepWtSum, edgePathDepWtSum, tagDepWtSum, setTagDepWtSum, setOvefitEdgePathDepWtSum
   

for sim in similarities:
    alledgePathDepWtSum, edgePathDepWtSum, tagDepWtSum, setTagDepWtSum, setOvefitEdgePathDepWtSum  = getDependencyPairSim(sim)
    df_features['ALl_EdgePath_Dep_Wt_by_' + sim[0:3]] = alledgePathDepWtSum
    df_features['EdgePath_Dep_Wt_by_' + sim[0:3]] = edgePathDepWtSum
    df_features['Tag_Dep_Wt_by_' + sim[0:3]] =  tagDepWtSum
    df_features['Set_Tag_Dep_Wt_by_' + sim[0:3]] = setTagDepWtSum
    df_features['Set_Ovefit_EdgePath_Dep_Wt_by_' + sim[0:3]] = setOvefitEdgePathDepWtSum
   

# storing features in txt file
df_features.to_csv('df_features_traint.txt',sep='\t',index=None)
#
##Split data into training and test sets
#
#
#df_model_f=df_features.copy()
#df_model_f.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
#df_model_f=df_model_f.drop(['id', 'Sentence1', 'Sentence2','Token_Sentence1',
#       'Token_Sentence2', 'Common_tokens', 'PosTag_Sentence1',
#       'PosTag_Sentence2', 'Lema_Sentence1', 'Lema_Sentence2',
#       'PosTagLema_Sentence1', 'PosTagLema_Sentence2'],axis=1)
##list_features=np.array([list(df_model_f['Normalized_NN']),
##                        list(df_model_f['Normalized_VB']),
##                        list(df_model_f['EdgePath_Dep_Wt']),
##                        list(df_model_f['Tag_Dep_Wt'])])
#list_features=np.array([
#                           
#                            list(df_model_f['Norm_NN_bypat']),
#                            list(df_model_f['Norm_VB_bypat']),
#                            list(df_model_f['Norm_NN_bylch']),
#                            list(df_model_f['Norm_VB_bylch']),
#                            list(df_model_f['Norm_NN_bywup']),
#                            list(df_model_f['Norm_VB_bywup']),
#                            list(df_model_f['Common_prepositions']),
#                           
#                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_pat']),
#                            list(df_model_f['EdgePath_Dep_Wt_by_pat']),
#                            list(df_model_f['Tag_Dep_Wt_by_pat']),
#                            list(df_model_f['Set_Tag_Dep_Wt_by_pat']),
#                            list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_pat']),
#                       
#                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_wup']),
#                            list(df_model_f['EdgePath_Dep_Wt_by_wup']),
#                            list(df_model_f['Tag_Dep_Wt_by_wup']),
#                            list(df_model_f['Set_Tag_Dep_Wt_by_wup']),
#                            list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_wup']),
#                       
#                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_lch']),
#                            list(df_model_f['EdgePath_Dep_Wt_by_lch']),
#                            list(df_model_f['Tag_Dep_Wt_by_lch']),
#                            list(df_model_f['Set_Tag_Dep_Wt_by_lch']),
#                            list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_lch']),
#                       
#                       
#                        ])
#list_features=np.transpose(list_features)
#X = list_features
#ylabels = list(df_model_f['Gold Tag'])
#
##
##def Majority_vote(X,ylabels):
##    import pickle
##    from sklearn.model_selection import train_test_split
##    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2,random_state=500,shuffle=False)
##   
##    ###Model_Building
##    #from sklearn import svm
##    from sklearn.ensemble import RandomForestClassifier
##    #from sklearn.linear_model import LinearRegression
##    from sklearn.linear_model import LogisticRegression
##    #
##    classifier=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
##    classifier=classifier.fit(X_train,y_train)
##   
##    #save model
##    pkl_filename = "Logistic_Regression.pkl"
##    with open(pkl_filename, 'wb') as file:
##        pickle.dump(classifier, file)
###    c=classifier.predict(X_test)
##   
##   
##    classifier1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
##    classifier1=classifier1.fit(X_train,y_train)
##   
##    #Save Model
##    pkl_filename1 = "Random_ForestClassifier.pkl"
##    with open(pkl_filename1, 'wb') as file:
##        pickle.dump(classifier1, file)
##   
###    c1=classifier1.predict(X_test)
##   
##   
###    from sklearn.svm import SVC
###    classifier2 = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
###    c2 = classifier2.predict(X_test)
##   
##    from sklearn.naive_bayes import MultinomialNB
##    classifier3 = MultinomialNB()
##    classifier3=classifier3.fit(X, ylabels)
##   
##    #Save Model
##    pkl_filename2 = "MultinomialNB.pkl"
##    with open(pkl_filename2, 'wb') as file:
##        pickle.dump(classifier3, file)
##
##    #Load models
##    with open(pkl_filename, 'rb') as file:
##        pickle_model = pickle.load(file)
##    c=pickle_model.predict(X_test)
##    with open(pkl_filename1, 'rb') as file:
##        pickle_model1 = pickle.load(file)
##    c1=pickle_model1.predict(X_test)
##    with open(pkl_filename2, 'rb') as file:
##        pickle_model2 = pickle.load(file)
##    c2=pickle_model2.predict(X_test)
##
##   
##   
##    mvotedlabel=[]
##    for ind in range(len(c)):
##        if c[ind]==c1[ind] and c[ind]==c2[ind]:
##            mvotedlabel.append(c[ind])
##        elif c[ind]==c1[ind] and c[ind]!=c2[ind]:
##            mvotedlabel.append(c[ind])      
##        elif c1[ind]==c2[ind] and c[ind]!=c1[ind]:
##            mvotedlabel.append(c1[ind])
##        elif c[ind]==c2[ind] and c[ind]!=c1[ind]:
##            mvotedlabel.append(c[ind])
##        else:
##            #Random Forest
##            mvotedlabel.append(c1[ind])
##   
##    from sklearn import metrics
##    accuracy=metrics.accuracy_score(y_test, c)
##    accuracy1=metrics.accuracy_score(y_test, c1)
##    accuracy2=metrics.accuracy_score(y_test, c2)
##   
##    print(accuracy,accuracy1,accuracy2)
##   
##    accuracy3=metrics.accuracy_score(y_test, mvotedlabel)
##    print("Majority Vote",accuracy3)
##
##
##Accuracy_mvoted=Majority_vote(X,ylabels)
