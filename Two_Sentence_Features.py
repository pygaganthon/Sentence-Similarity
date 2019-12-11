#Gagandeep Singh Jossan gxj170003
#Maitreyee Mhasakar mam171630

import pandas as pd
import numpy as np


def predictGoldTag(Sentence1,Sentence2):
    import pandas as pd
    from nltk.corpus import wordnet as wn
    import spacy
    nlp = spacy.load('en')
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk import pos_tag
    import numpy as np
    from nltk.corpus.reader import NOUN
    from nltk.corpus.reader import VERB
   
       
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
   
   
   
    from nltk.tokenize import word_tokenize
    Token_Sentence1 = word_tokenize(Sentence1)
    Token_Sentence2 = word_tokenize(Sentence2)

    from nltk.corpus import wordnet
    listOfSSHypePair1 = []
    listOfSSHypoPair1 = []
    listOfSSHoloPair1 = []
    listOfSSMeroPair1 = []
    for word in Token_Sentence1:
        for ss in wordnet.synsets(word):
            listOfSSHypePair1.extend(ss.hypernyms())
            listOfSSHypoPair1.extend(ss.hyponyms())
            listOfSSHoloPair1.extend(ss.part_holonyms())
            listOfSSMeroPair1.extend(ss.part_meronyms())
    
    listOfSSHypePair2 = []
    listOfSSHypoPair2 = []
    listOfSSHoloPair2 = []
    listOfSSMeroPair2 = []
    for word in Token_Sentence2:
        for ss in wordnet.synsets(word):
            listOfSSHypePair2.extend(ss.hypernyms())
            listOfSSHypoPair2.extend(ss.hyponyms())
            listOfSSHoloPair2.extend(ss.part_holonyms())
            listOfSSMeroPair2.extend(ss.part_meronyms())
   
    common_token=set(Token_Sentence1) and set(Token_Sentence2)
   
   
    postag_Sentence1={}
    postag_Sentence2={}

    sen1temp=[]
    sen2temp=[]
    for i in Token_Sentence1:
        sen1temp.append(i)
    for j in Token_Sentence2:
        sen2temp.append(j)
    postag_Sentence1 = dict(pos_tag(sen1temp))
    postag_Sentence2 = dict(pos_tag(sen2temp))
       
    lemmatizer = WordNetLemmatizer()
    lemmatized_Sentence1=[]
    lemmatized_Sentence2=[]



    for idx,i in enumerate(Token_Sentence1):
#        print(penn_to_wn(df_features['PosTag_Sentence1'][index][idx][1]))
        w1 =lemmatizer.lemmatize(i,penn_to_wn(postag_Sentence1[i],True))
        lemmatized_Sentence1.append(w1)
    for idx,j in enumerate(Token_Sentence2):
#        print(penn_to_wn(df_features['PosTag_Sentence2'][index][idx][1]))
        w2 =lemmatizer.lemmatize(j,penn_to_wn(postag_Sentence2[j],True))
        lemmatized_Sentence2.append(w2)
   
    postaglemma_Sentence1 = dict(pos_tag(sen1temp))
    postaglemma_Sentence2 = dict(pos_tag(sen2temp))
   



    lema_token_s1 = []
    lema_token_s2 = []
   
   

    doc1 = nlp(Sentence1)
    for token in doc1:
        lemw1 = lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True))
        lema_token_s1.append(lemw1)
       
    tknsen1ptags = dict(pos_tag(lema_token_s1))
#    lemmatizer.lemmatize(token.head.text,penn_to_wn(token.head.tag_,True)),

    doc2 = nlp(Sentence2)
    for token in doc2:
        lemw2 = lemmatizer.lemmatize(token.text,penn_to_wn(token.tag_,True))
        lema_token_s2.append(lemw2)
       
    tknsen2ptags = dict(pos_tag(lema_token_s2))

       
#    df_features["Token_Lema_Sent_1"] = lema_token_s1
#    df_features["Token_Lema_Sent_2"] = lema_token_s2
#    df_features["Token_LemaPOS_Sent_1"] = tknsen1ptags
#    df_features["Token_LemaPOS_Sent_2"] = tknsen2ptags
   

    from nltk import pos_tag

    #List to store nouns and verbs in common in both sentences/total number of nouns and verbs in the sentence repectively.
    Common_nouns_sen12=[]
    Common_verbs_sen12=[]

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
                except:
                    val_sim = -1
                if(val_sim == None or val_sim == -1 or val_sim < 0.01):
                    if(sim=="path_similarity"):
                        val_sim = 0.1
                    if(sim=="lch_similarity"):
                        val_sim = 1.2
                    if(sim=="wup_similarity"):
                        val_sim = 0.1
                lst.append(val_sim)
        if(len(lst)==0):
            if(sim=="path_similarity"):
                val_sim = 0.1
            if(sim=="lch_similarity"):
                val_sim = 1.2
            if(sim=="wup_similarity"):
                val_sim = 0.1
            lst.append(val_sim)
        return pd.Series(lst).dropna()
   
#    HyperCommonDone = False
    def getNormalizedCount(sim):
        Common_nouns_sen12=[]
        Common_verbs_sen12=[]
       
        list_of_nouns_sen1=[]
        list_of_verbs_sen1=[]
     
        list_of_nouns_sen2=[]
        list_of_verbs_sen2=[]
 
        #Variables to store count of number of nouns and verbs in the sentence.
        common_nouns=0
        common_verbs=0
    #Generate list of Nouns and Verbs in Sentence 1 and 2.
     
        for i in pos_tag(lemmatized_Sentence1):
    #        if i[1] in Nouns:
            if penn_to_wn(i[1],False)=='n':
                #Actual lematized token is appended to the list
                list_of_nouns_sen1.append(i[0])
            elif penn_to_wn(i[1],False)=='v':
                list_of_verbs_sen1.append(i[0])
 
        for j in pos_tag(lemmatized_Sentence2):
    #        if j[1] in Nouns:
            if penn_to_wn(j[1],False)=='n':
                list_of_nouns_sen2.append(j[0])
 
    #        elif j[1] in Verbs:
            elif penn_to_wn(j[1],False)=='v':
                list_of_verbs_sen2.append(j[0])
 
 
        #Compare nouns from sentence 1 and 2
#        if HyperCommonDone != True:
#            score_common_hyp = 0
#            lstcm_hypernyms = []
        for word in list_of_nouns_sen1:
            for wordsen2 in list_of_nouns_sen2:
               
#                if HyperCommonDone != True:
#                    try:
#                        wordsyns = wn.synset(str(word) + '.n.01')
#                        wordsens2syns = wn.synset(str(wordsen2) + '.n.01')
#                        common_hyper_2words = wordsyns.lowest_common_hypernyms(wordsens2syns)
#                        lstcm_hypernyms.extend(common_hyper_2words)
#                        for hyp in common_hyper_2words:
#                            score_common_hyp = score_common_hyp + hyp.min_depth()
#                    except:
#                        score_common_hyp = score_common_hyp
#                        
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
               
#                if HyperCommonDone != True:
#                    try:
#                        wordsyns = wn.synset(str(word1) + '.v.01')
#                        wordsens2syns = wn.synset(str(word1sen2) + '.v.01')
#                        common_hyper_2words = wordsyns.lowest_common_hypernyms(wordsens2syns)
#                        lstcm_hypernyms.extend(common_hyper_2words)
#                        for hyp in common_hyper_2words:
#                            score_common_hyp = score_common_hyp + hyp.min_depth()
#                    except:
#                        score_common_hyp = score_common_hyp
                       
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
 
#        if HyperCommonDone != True:              
#            if(score_common_hyp==0):
#                comon_hyper_score.append(0)
#            else:
#                comon_hyper_score.append(score_common_hyp / len(set(lstcm_hypernyms)))
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
        return Common_nouns_sen12_new, Common_verbs_sen12_new
     
   

    #                , "res_similarity", "jcn_similarity", "lin_similarity"]
   
   
    Norm_NN_bypat, Norm_VB_bypat = getNormalizedCount("path_similarity")
#    if HyperCommonDone != True:
#        HyperCommonDone = True
    Norm_NN_bylch, Norm_VB_bylch = getNormalizedCount("lch_similarity")
    Norm_NN_bywup, Norm_VB_bywup = getNormalizedCount("wup_similarity")
   
   
   
   
#    prep={'of':1,'in':2,'to':3,'for':4,'with':5,'on':6,'at':7,'from':8,'by':9,'about':10,'as':11,'into':12,'like':13,'through':14,'after':15,'over':16,'between':17,'out':18,'against':19,'during':20,'without':21,'before':22,'under':23,'around':24,'among':25}
#    def Prep(Token_Sentence1, Token_Sentence2):
#        List_common_prep=[]
#        sen1_prep=[]
#        sen2_prep=[]
#
#        for i in Token_Sentence1:
#            if i in prep:
#                sen1_prep.append(i)
#
#        for j in Token_Sentence2:
#            if j in prep:
#                sen2_prep.append(j)
#        Common_prep=[]
#
#        for k in sen1_prep:
#            if k in sen2_prep:
#                Common_prep.append(k)
#        List_common_prep.append((len(Common_prep)/25)*100)
#        return List_common_prep
       
       
   

#    Common_prepositions= Prep(Token_Sentence1, Token_Sentence2)


    lemmatizer = WordNetLemmatizer()
    verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
    noun = ['NN','NNS','NNP','NNPS']
    nounverb = ['VB','VBD','VBG','VBN','VBP','VBZ','NN','NNS','NNP','NNPS']

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
       
        tagrelations1 = []
        wordrelations1 = []
        allwordrelations1 = []
       
        doc1 = nlp(Sentence1)
        root1=[]
        dep_s1 = []
        # record all possible edges
        for token in doc1:
            dep_s1.append((token.head , token.dep_,token))
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
        doc2 = nlp(Sentence2)
        
        dep_s2 = []
        
         # record all possible edges
        for token in doc2:
            dep_s2.append((token.head , token.dep_,token))
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
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(tknsen2ptags[wr1[0]]), returnCategory(tknsen1ptags[wr2[0]]))
                    else:
#                        print(wr2[0], wr1[0] , df_features["Token_LemaPOS_Sent_2"][index][wr2[0]],df_features["Token_LemaPOS_Sent_1"][index][wr1[0]])
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(tknsen1ptags[wr1[0]]), returnCategory(tknsen2ptags[wr2[0]]))
                    if len(tmp)>0:
                        h = tmp.max()
                    else:
                        h = 0.1
                if(wr1[1]==wr2[1]):
                    maxd = 1
                else:
                    if(swapped):
#                        print(wr1[1], df_features["PosTagLema_Sentence2"][index][wr1[1]], wr2[1], df_features["PosTagLema_Sentence1"][index][wr2[1]])
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(tknsen2ptags[wr1[1]]), returnCategory(tknsen1ptags[wr2[1]]))
                    else:
#                        print(wr1[1], df_features["PosTagLema_Sentence1"][index][wr1[1]], wr2[1], df_features["PosTagLema_Sentence2"][index][wr2[1]])
#                        print(wr1[1] , wr2[1])
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(tknsen1ptags[wr1[1]]), returnCategory(tknsen2ptags[wr2[1]]))
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
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(tknsen2ptags[wr1[0]]), returnCategory(tknsen1ptags[wr2[0]]))
                    else:
                        tmp = getSimilarity(sim, wr1[0], wr2[0], returnCategory(tknsen1ptags[wr1[0]]), returnCategory(tknsen2ptags[wr2[0]]))
                    if len(tmp)>0:
                        h = tmp.max()
                    else:
                        h = 0
                if(wr1[1]==wr2[1]):
                    maxd = 1
                else:
                    if(swapped):
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(tknsen2ptags[wr1[1]]), returnCategory(tknsen1ptags[wr2[1]]))
                    else:
                        tmp = getSimilarity(sim, wr1[1], wr2[1], returnCategory(tknsen1ptags[wr1[1]]), returnCategory(tknsen2ptags[wr2[1]]))
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
        return alledgePathDepWtSum, edgePathDepWtSum, tagDepWtSum, setTagDepWtSum, setOvefitEdgePathDepWtSum, dep_s1, dep_s2

    alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat, dep_s1, dep_s2  = getDependencyPairSim("path_similarity")
    alledgePathDepWtSumbylch, edgePathDepWtSumbylch, tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch, dep_s1, dep_s2  = getDependencyPairSim("lch_similarity")
    alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup, dep_s1, dep_s2  = getDependencyPairSim("wup_similarity")
    output=[Norm_NN_bypat, Norm_VB_bypat, Norm_NN_bylch, Norm_VB_bylch, Norm_NN_bywup, Norm_VB_bywup, alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat, alledgePathDepWtSumbylch, edgePathDepWtSumbylch, tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch, alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup, listOfSSHypePair1, listOfSSHypoPair1, listOfSSHoloPair1, listOfSSMeroPair1, listOfSSHypePair2, listOfSSHypoPair2, listOfSSHoloPair2, listOfSSMeroPair2, Token_Sentence1, Token_Sentence2, lemmatized_Sentence1, lemmatized_Sentence2, postag_Sentence1, postag_Sentence2, dep_s1, dep_s2]
#    return Norm_NN_bypat, Norm_VB_bypat, Norm_NN_bylch, Norm_VB_bylch, Norm_NN_bywup, Norm_VB_bywup, alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat, alledgePathDepWtSumbylch, edgePathDepWtSumbylch, tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch, alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup, listOfSSHypePair1, listOfSSHypoPair1, listOfSSHoloPair1, listOfSSMeroPair1, listOfSSHypePair2, listOfSSHypoPair2, listOfSSHoloPair2, listOfSSMeroPair2, Token_Sentence1, Token_Sentence2, lemmatized_Sentence1, lemmatized_Sentence2, postag_Sentence1, postag_Sentence2, dep_s1, dep_s2
    return output

#Norm_NN_bypat, Norm_VB_bypat, Norm_NN_bylch, Norm_VB_bylch, Norm_NN_bywup, Norm_VB_bywup, comon_hyper_score,  Common_prepositions, alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat, alledgePathDepWtSumbylch, edgePathDepWtSumbylch, tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch, alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup = predictGoldTag(Sentence1,Sentence2)

#            listOfSSHypePair1, listOfSSHypoPair1, listOfSSHoloPair1, listOfSSMeroPair1, listOfSSHypePair2, listOfSSHypoPair2, listOfSSHoloPair2, listOfSSMeroPair2, Token_Sentence1, Token_Sentence2, lemmatized_Sentence1, lemmatized_Sentence2, postag_Sentence1, postag_Sentence2, dep_s1, dep_s2

def Majority_vote(X_train,X_test,y_train):
    import pickle
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2,random_state=500,shuffle=False)
    
    ###Model_Building
    #from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    #
    classifier=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
    classifier=classifier.fit(X_train,y_train)
    
    #save model
    pkl_filename = "Logistic_Regression.pkl"
#    with open(pkl_filename, 'wb') as file:
#        pickle.dump(classifier, file)
    c=classifier.predict(X_test)
    
    
    classifier1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    classifier1=classifier1.fit(X_train,y_train)
    c1=classifier1.predict(X_test)

    
    #Save Model
#    pkl_filename1 = "Random_ForestClassifier.pkl"
#    with open(pkl_filename1, 'wb') as file:
#        pickle.dump(classifier1, file)
    
    
    
#    from sklearn.svm import SVC 
#    classifier2 = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
#    c2 = classifier2.predict(X_test)
    
    from sklearn.naive_bayes import MultinomialNB
    classifier3 = MultinomialNB()
    classifier3=classifier3.fit(X_train, y_train)
    c2=classifier3.predict(X_test)

    
    #Save Model
#    pkl_filename2 = "MultinomialNB.pkl"
#    with open(pkl_filename2, 'wb') as file:
#        pickle.dump(classifier3, file)

    #Load models
#    with open(pkl_filename, 'rb') as file:
#        pickle_model = pickle.load(file)
#    c=pickle_model.predict(X_test)
#    with open(pkl_filename1, 'rb') as file:
#        pickle_model1 = pickle.load(file)
#    c1=pickle_model1.predict(X_test)
#    with open(pkl_filename2, 'rb') as file:
#        pickle_model2 = pickle.load(file)
#    c2=pickle_model2.predict(X_test)

    
    
    mvotedlabel=[]
    for ind in range(len(c)):
        if c[ind]==c1[ind] and c[ind]==c2[ind]:
            mvotedlabel.append(c[ind])
        elif c[ind]==c1[ind] and c[ind]!=c2[ind]:
            mvotedlabel.append(c[ind])       
        elif c1[ind]==c2[ind] and c[ind]!=c1[ind]:
            mvotedlabel.append(c1[ind])
        elif c[ind]==c2[ind] and c[ind]!=c1[ind]:
            mvotedlabel.append(c[ind])
        else:
            #Random Forest
            mvotedlabel.append(c1[ind])
    
#    from sklearn import metrics
#    accuracy=metrics.accuracy_score(y_test, c)
#    accuracy1=metrics.accuracy_score(y_test, c1)
#    accuracy2=metrics.accuracy_score(y_test, c2)
#    
#    print(accuracy,accuracy1,accuracy2)
#    
#    accuracy3=metrics.accuracy_score(y_test, mvotedlabel)
#    print("Majority Vote",accuracy3)
#    print(c,c1,c2)
    
    return int(c1)




def predict_sentence(X_test):
    df=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//data/train-set.txt', sep="\t",error_bad_lines=False)
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

    #df=pd.read_csv('C:/Users/gagan/OneDrive/Desktop/NLPProject/data/sample_train.txt', sep="\t",error_bad_lines=False)
    df_features=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//df_features_train//df_features_traintCopy.txt', sep="\t",error_bad_lines=False)

    df_features.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df_model_f=df_features.copy()
    df_model_f.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df_model_f=df_model_f.drop(['id', 'Sentence1', 'Sentence2','Token_Sentence1',
                            'Hyper_Sentence1','Hyper_Sentence2','Hypo_Sentence1',
                            'Hypo_Sentence2','Holo_Sentence1','Holo_Sentence2',
                            'Mero_Sentence1','Mero_Sentence2','Token_Sentence2',
                            'Common_tokens', 'PosTag_Sentence1',
       'PosTag_Sentence2', 'Lema_Sentence1', 'Lema_Sentence2',
       'PosTagLema_Sentence1', 'PosTagLema_Sentence2','EdgePath_Dep_Wt_by_lch','Common_Hyper_Score','Common_prepositions','ALl_EdgePath_Dep_Wt_by_lch'],axis=1)

    list_features=np.array([
                               
                                list(df_model_f['Norm_NN_bypat']),
                                list(df_model_f['Norm_VB_bypat']),
                                list(df_model_f['Norm_NN_bylch']),
                                list(df_model_f['Norm_VB_bylch']),
                                list(df_model_f['Norm_NN_bywup']),
                                list(df_model_f['Norm_VB_bywup']),
    #                            list(df_model_f['Common_prepositions']),
    #                            list(df_model_f["Common_Hyper_Score"]),
                               
                                list(df_model_f['ALl_EdgePath_Dep_Wt_by_pat']),
                                list(df_model_f['EdgePath_Dep_Wt_by_pat']),
                                list(df_model_f['Tag_Dep_Wt_by_pat']),
                                list(df_model_f['Set_Tag_Dep_Wt_by_pat']),
                                list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_pat']),
                           
                                list(df_model_f['ALl_EdgePath_Dep_Wt_by_wup']),
                                list(df_model_f['EdgePath_Dep_Wt_by_wup']),
                                list(df_model_f['Tag_Dep_Wt_by_wup']),
                                list(df_model_f['Set_Tag_Dep_Wt_by_wup']),
                                list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_wup']),
                           
    #                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_lch']),
    #                            list(df_model_f['EdgePath_Dep_Wt_by_lch']),
                                list(df_model_f['Tag_Dep_Wt_by_lch']),
                                list(df_model_f['Set_Tag_Dep_Wt_by_lch']),
                                list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_lch']),
                           
                           
                            ])
    list_features=np.transpose(list_features)
    X_train=list_features
    y_train=list(df_model_f['Gold Tag'])
    predicted_label=Majority_vote(X_train,X_test,y_train)
    return predicted_label



def main():
    Sentence1 = input()
    Sentence2 = input()
    Norm_NN_bypat, Norm_VB_bypat, Norm_NN_bylch, Norm_VB_bylch, Norm_NN_bywup, Norm_VB_bywup, alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat, alledgePathDepWtSumbylch, edgePathDepWtSumbylch, tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch, alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup, listOfSSHypePair1, listOfSSHypoPair1, listOfSSHoloPair1, listOfSSMeroPair1, listOfSSHypePair2, listOfSSHypoPair2, listOfSSHoloPair2, listOfSSMeroPair2, Token_Sentence1, Token_Sentence2, lemmatized_Sentence1, lemmatized_Sentence2, postag_Sentence1, postag_Sentence2, dep_s1, dep_s2=predictGoldTag(Sentence1,Sentence2)
    #print(predictGoldTag(Sentence1,Sentence2))
    print(Norm_NN_bypat, Norm_VB_bypat, Norm_NN_bylch, Norm_VB_bylch, Norm_NN_bywup, Norm_VB_bywup, alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat, alledgePathDepWtSumbylch, edgePathDepWtSumbylch, tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch, alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup, listOfSSHypePair1, listOfSSHypoPair1, listOfSSHoloPair1, listOfSSMeroPair1, listOfSSHypePair2, listOfSSHypoPair2, listOfSSHoloPair2, listOfSSMeroPair2, Token_Sentence1, Token_Sentence2, lemmatized_Sentence1, lemmatized_Sentence2, postag_Sentence1, postag_Sentence2, dep_s1, dep_s2)
    
    
    
    
    X_test=np.array([Norm_NN_bypat, Norm_VB_bypat, Norm_NN_bylch, Norm_VB_bylch, Norm_NN_bywup, Norm_VB_bywup, alledgePathDepWtSumbypat, edgePathDepWtSumbypat, tagDepWtSumbypat, setTagDepWtSumbypat, setOvefitEdgePathDepWtSumbypat,alledgePathDepWtSumbywup, edgePathDepWtSumbywup, tagDepWtSumbywup, setTagDepWtSumbywup, setOvefitEdgePathDepWtSumbywup,tagDepWtSumbylch, setTagDepWtSumbylch, setOvefitEdgePathDepWtSumbylch])
    X_test=np.transpose(X_test)
    plabel=predict_sentence(X_test)
    print("Predicted Label",plabel)
   
    
if __name__ == "__main__":
    main()
       
