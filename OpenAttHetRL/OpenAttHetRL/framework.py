#atthet 
# ranking case study:  expert finding in question answering system
import numpy as np
import tensorflow as tf
from networkx import to_numpy_matrix
import networkx as nx
import datetime
import sys
import os
import pickle
try:
    import ujson as json
except:
    import json
import math
from scipy.linalg import fractional_matrix_power
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math

class AttHetRL:  
    def  __init__(self,data,dim_node=32,dim_word=300,epoch=10,batch_size=16):        
        self.dataset=data 
        self.parsed="/parsed"
        self.epochs=epoch
        self.batch_size=batch_size
        self.d=dim_node #embeding dim
        self.embedding_size=dim_word
        
    def init_model(self):        
        self.loadG()        
        self.GCNW_1 =AttHetRL.weight_variable((self.G.number_of_nodes(), 32)) 
        print("GCNW1=",self.GCNW_1.shape)
        self.GCNW_2 =AttHetRL.weight_variable((self.GCNW_1.shape[1], self.d))
        print("GCNW2=",self.GCNW_2.shape)
        
        #regression layer
        self.regindim=4*self.GCNW_2.shape[1]+11
        self.W1=AttHetRL.weight_variable((self.regindim,8))
        #self.W2=EndCold.weight_variable((self.W1.shape[1],8))
        #self.W3=EndCold.weight_variable((self.W2.shape[1],16))
        self.W4 = AttHetRL.weight_variable2(self.W1.shape[1])
        #self.W4 = EndCold.weight_variable2(4*self.GCNW_2.shape[1])
        self.b = tf.Variable(random.uniform(0, 1))
        self.inputs=[]
        self.outputs=[]     
    
        self.n_bins=11                
        self.lamb = 0.5

        self.mus = AttHetRL.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = AttHetRL.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        
        self.embeddings = tf.Variable(tf.random.uniform([self.vocab_size+1, self.embedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
       
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0]+shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)
    
    def weight_variable2(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape)
        initial = tf.random.uniform([shape,1], minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)
    
    def prepare_train_test_data(self):
        train_data=[]  #
        train_label=[] # 
        INPUT=self.dataset+self.parsed+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        self.N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        self.qnum=int(line.split(" ")[2])
        line=pfile.readline()
        self.usernum=int(line.split(" ")[2])
        #line=pfile.readline()
        #self.answerernum=int(line.split(" ")[2])
        line=pfile.readline()
        self.tagnum=int(line.split(" ")[2])
        
        self.Q_id_map={}
        INPUT2=self.dataset+self.parsed+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            self.Q_id_map[e[1]]=e[0]
        #print(self.Q_id_map)
       
        self.user_id_map={}
        INPUT3=self.dataset+self.parsed+"/user_id_map.txt"
#         ids=np.loadtxt(INPUT3, dtype=int)
#         for e in ids:
#             self.user_id_map[e[1]]=self.qnum+e[0]
        
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=uname.strip()
            self.user_id_map[uname]=self.qnum+int(e[0])            
            line=fin.readline().strip()
            
        fin.close()
        
        #print(self.asker_id_map)
        
        #self.answerer_id_map={}
        #INPUT=self.dataset+"/ColdEndFormat/answerer_id_map.txt"
        #ids=np.loadtxt(INPUT, dtype=int)
        #for e in ids:
        #    self.answerer_id_map[e[1]]=self.qnum+self.askernum+e[0]
        #print( self.answerer_id_map)
        
        self.tag_id_map={}
        INPUT4=self.dataset+self.parsed+"/tag_id_map.txt"
        with open( INPUT4, "r") as fin:                
            for line in fin:
                data = line.strip().split(" ")        
                self.tag_id_map[data[1]]=self.qnum+self.usernum+int(data[0])
        #print( self.tag_id_map) 
        
        self.Answer_score_map={}
        INPUT6=self.dataset+self.parsed+"/A_score.txt"
        ids=np.loadtxt(INPUT6, dtype=int)
        for e in ids:
            self.Answer_score_map[e[0]]=e[1]
           
        INPUT7=self.dataset+self.parsed+"/Record_Train.json"        
        with open(INPUT7, "r",encoding="utf8") as fin:
            for line in fin:
                data = json.loads(line)
                
                qid = int(data['QuestionId'])
                qmapedid=[self.Q_id_map[qid]]
                
                qaskerid = data['QuestionOwnerId']
                qaskermapid=[self.user_id_map[qaskerid]]
                
                qtags=data['Tags']
                qtagslist=[self.tag_id_map[qtags[0]]]
                for qtag in qtags[1:]:
                    qtagslist.append(self.tag_id_map[qtag])
                
                answerers=data["AnswererAnswerTuples"]
                for answerer in answerers:
                    answererid=answerer[0]
                    answerid=int(answerer[1])
                    qanswerermapid=[self.user_id_map[answererid]]
                    score=self.Answer_score_map[answerid]
                    item=np.concatenate((qmapedid,qaskermapid,qanswerermapid,[answerid],qtagslist))                    
                    train_data.append(item)
                    train_label.append(score)
          
        train_data=np.array(train_data)
        print(train_data)
        OUTPUT8=self.dataset+self.parsed+"/"+"train_data.txt"
        fout_train=open(OUTPUT8,"w")
        OUTPUT9=self.dataset+self.parsed+"/train_labels.txt"
        fout_label=open(OUTPUT9,"w")
         
        # add negative samples for train data
        
        # end add negative data
        
        
        for ii in range(len(train_data)):
            strdata=""
            for data in train_data[ii]:
                strdata+=str(data)+" "
            fout_train.write(strdata.strip()+"\n")
            fout_label.write(str(train_label[ii])+"\n")
            
        fout_train.close()
        fout_label.close()
        
        
        self.record_all_data={}
        self.u_answers={}
        INPUT=self.dataset+self.parsed+"/Record_All.json"
        unknown=0
        with open(INPUT, 'r',encoding="utf8") as fin_all:
            for line in fin_all:
                data = json.loads(line)                
                qid = data.get('QuestionId')
                QOwnerId=data.get('QuestionOwnerId')
                AccAnswerId=data.get('AcceptedAnswerId')
                AccAnswererId=data.get('AcceptedAnswererId')
                AnswererIdList=data.get('AnswererIdList')                                   
                AnswererAnswerTuples=data.get('AnswererAnswerTuples')  
                
                for aa in AnswererAnswerTuples:
                    uid=aa[0]
                    aid=aa[1]
                    score=self.Answer_score_map[int(aid)]
                    if uid not in  self.u_answers:
                        self.u_answers[uid]=[]
                    self.u_answers[uid].append([aid,score])    
                
                self.record_all_data[qid]={}       
                self.record_all_data[qid]['QuestionOwnerId']=QOwnerId
                self.record_all_data[qid]['AcceptedAnswerId']=AccAnswerId
                self.record_all_data[qid]['AcceptedAnswererId']=AccAnswererId
                self.record_all_data[qid]['AnswererIdList']=AnswererIdList
                self.record_all_data[qid]['AnswererAnswerTuples']=AnswererAnswerTuples
                self.record_all_data[qid]['Tags']=data.get('Tags')
                
        
        OUTPUT=self.dataset+self.parsed+"/user_answers.txt"        
        fout=open(OUTPUT,"w")
        for u in self.u_answers:
            fout.write(u) 
            for aa in self.u_answers[u]:
                fout.write(" "+aa[0]+" "+str(aa[1]))
            fout.write("\n")    
        fout.close()
        self.Q_tags={}
        INPUT10=self.dataset+self.parsed+"/Q_tags.txt"        
        with open(INPUT10, "r") as fin:
            for line in fin:
                elem=line.strip().split(" ")
                self.Q_tags[int(elem[0])]=elem[1:]
        
        
        
        answerers=[]
        INPUT=self.dataset+self.parsed+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answerers.append(int(d[0]))
                
        test_data=[]
        INPUT=self.dataset+self.parsed+"/test.txt"        
        with open(INPUT, "r") as fin1:
            for line in fin1:                
                elem=line.strip().split(" ")
                
                qaskerid=elem[0]
                qid=int(elem[1])                
                qmapid=[self.Q_id_map[qid]]                
                qaskeremapid=[self.user_id_map[qaskerid]]
                qtags=self.Q_tags[qid]
                qtagslist=[self.tag_id_map[qtags[0]]]
                for qtag in qtags[1:]:
                    qtagslist.append(self.tag_id_map[qtag])
                item=np.concatenate((qmapid,qaskeremapid,qtagslist))                
                AATuples=self.record_all_data[str(qid)]['AnswererAnswerTuples']
                
                answererlst=[]
                posids=[]
                for aa in AATuples:
                    arid=aa[0]
                    posids.append(aa[0])
                    answerid=aa[1]
                    score=self.Answer_score_map[int(answerid)]
                    answererlst.append([arid,answerid,score])
                #answererlst=np.array(answererlst)
                
                
                lenaa=len(answererlst)
                neg_answererlst=[]                
                
               
                #add negetive samples
                for i in range(lenaa):
                    neid=random.choice(answerers)
                    while neid in posids:
                        neid=random.choice(answerers)
                    neg_answererlst.append([neid,-1,0])
                answererlst.extend(neg_answererlst)
                test_data.append([[item],answererlst])
        
        test_data=np.array(test_data)
        print(test_data)
        
        OUTPUT=self.dataset+self.parsed+"/test_data.txt"        
        fout_test=open(OUTPUT,"w")        
         
        for ii in range(len(test_data)):
            strdata=""
            for data in test_data[ii][0]:
                for d in data:
                    strdata+=str(d)+" "
            strdata=strdata.strip()
            strdata+=";"
            for data in test_data[ii][1]:
                strdata+=str(data[0])+" "+str(data[1])+" "+str(data[2])+" "
            fout_test.write(strdata.strip()+"\n")      
        fout_test.close()
        
        print("prepare data done!!")
      
    def load_traindata(self,qlen,alen):
        self.train_data=[]
        self.train_label=[]
        
        
        INPUT=self.dataset+self.parsed+"/train_data.txt"
        fin_train=open(INPUT)
        INPUT2=self.dataset+self.parsed+"/train_labels.txt"
        fin_label=open(INPUT2)
        train=fin_train.readline().strip()
        label=fin_label.readline().strip()
        while train:
            data=train.split(" ")
            lst=[]
            for d in data:
                lst.append(int(d)) 
            self.train_data.append(lst)
            train=fin_train.readline().strip()
            datal=float(label)
            self.train_label.append(datal)
            label=fin_label.readline().strip()
        fin_train.close()
        fin_label.close()
        self.train_data=np.array(self.train_data)
        
        #self.train_label=np.array(self.train_label)
        
        #add nagetive samples
        INPUT=self.dataset+self.parsed+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        self.N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])     
        user_id_map={}
        INPUT3=self.dataset+self.parsed+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=int(uname.strip())
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
        fin.close() 
        
        
        answerers=[]
        INPUT=self.dataset+self.parsed+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answerers.append(int(d[0]))
        new_data=list(self.train_data)        
        ids=np.array([self.train_data[i][0] for i in range(self.train_data.shape[0])])
        for i in set(ids): 
            #print(i)
            ind=np.where(ids==i)
            answerer_posids=[ a[2] for a in self.train_data[ind]]
            #print(answerer_posids)
            qaetinfo=self.train_data[ind][0].copy()
            #print(qaetinfo)
            qaetinfo[3]=-1
            for kk in range(len(answerer_posids)):
                neid=user_id_map[random.choice(answerers)]
                while neid in answerer_posids:
                    neid=user_id_map[random.choice(answerers)]
                #qaetinfo[2]=neid
                p1=qaetinfo[0:2].copy()
                p1.append(neid)
                p1.extend(qaetinfo[3:])
                new_data.append(p1)
                self.train_label.append(0)
                
            
        self.train_data=np.array(new_data)
        self.train_label=np.array(self.train_label)
        #print(self.train_data[-10:])
        #sys.exit(0)
        #end nagetive
        
        # normalize scores
        ids=np.array([self.train_data[i][0] for i in range(self.train_data.shape[0])])
        #print(ids[:10])
        norm_lbls=np.zeros(len(self.train_label))
        
        for i in set(ids): 
            #print(i)
            ind=np.where(ids==i)
            #print(ind)
            #print(self.train_label[ind])
            minscoe=min(self.train_label[ind])
            if minscoe<0:
                #print(self.train_label[ind])
                self.train_label[ind]=self.train_label[ind]+(-1.0*minscoe)
                #print(self.train_label[ind])
            maxscoe=max(self.train_label[ind])
            if maxscoe==0:
                self.train_label[ind]+=1
            #print(self.train_label[ind]) 
            #print(5*(self.train_label[ind]/np.sum(self.train_label[ind]))+2)
            norm_lbls[ind]=20*(self.train_label[ind]/np.sum(self.train_label[ind]))+5
            #print(20*(self.train_label[ind]/np.sum(self.train_label[ind]))+5)
        #print(norm_lbls[:10])
        #print(self.train_label[:20])
        #norm_lbls=np.array(norm_lbls)
        #ind=np.where(self.train_label!=0)
        #print(self.val_label[:20])
        #self.train_label[ind]=norm_lbls[ind]
        
        self.train_label=np.array(norm_lbls)
        
        #print(self.train_label[:20])
        #sys.exit(0)       
        #shuffle
        ind_new=[i for i in range(len(self.train_data))]
        np.random.shuffle(ind_new)
        self.train_data=self.train_data[ind_new,]
        self.train_label=self.train_label[ind_new,]
        
        # load q and answer text
        
        self.qatext=[]
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        
        INPUT=self.dataset+self.parsed+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=self.dataset+self.parsed+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=self.dataset+self.parsed+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=self.dataset+self.parsed+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:]        
        
        Q_id_map={}
        INPUT2=self.dataset+self.parsed+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map[int(e[0])]=int(e[1])
        
        u_answers={}
        INPUT=self.dataset+self.parsed+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[user_id_map[int(d[0])]]=d[1::2]
        
        self.max_q_len=qlen
        self.max_d_len=alen
        self.vocab_size=len(vocab)
        
        delindx=0
        delindexes=[]
        for td in self.train_data:
            #print(td)
            qid=Q_id_map[td[0]]
            #print(qid)
            
            aid=td[3] 
            #print(aid)
            qtext=qtitle[qid].copy()
            qtext.extend(qcontent[qid])            
            qtext=qtext[:self.max_q_len]
            #print(qtext)
            qt=[]
            for wr in qtext:
                qt.append(vocab.index(wr)+1)
            padzeros=self.max_q_len-len(qt)
            #for zz in range(padzeros):
                 #qt.append(0)
            if aid!=-1:        
                atext=answers[aid]
                atext=atext[:self.max_d_len]
                #print(atext)
                at=[]
                for wr in atext:
                    if wr in vocab:
                        at.append(vocab.index(wr)+1)
                    else:
                        print(str(wr)+" not in  vocab" )

                padzeros=self.max_d_len-len(at)
                #for zz in range(padzeros):
                #     at.append(0)
            else:
                e=td[2]
                etext1=[]
                for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                etext=etext1
                    #etext=etext1
                if len(etext1)>self.max_d_len:                         
                        etext=random.sample(etext1,self.max_d_len)
                        
                
                #print(etext)
                etext2=[]
                for ii in range(len(etext)):
                    if etext[ii] in vocab:
                        etext2.append(vocab.index(etext[ii])+1)
                    else:
                        print(str(etext[ii])+" not in  vocab" )
                at=etext2.copy()    
            #self.qatext.append([qt,at])
            if len(qt)==0 or len(at)==0:
                delindexes.append(delindx)
            else: 
                self.qatext.append([qt,at])
            delindx+=1    
        self.qatext=np.array(self.qatext) 
#         print(self.qatext[:3])
#         print(self.train_data[:3])
#         print(delindexes)
        if len(delindexes)!=0: #remove q with no answer
            self.train_data=np.delete(self.train_data,delindexes)
            self.train_label=np.delete(self.train_label, delindexes)
            #self.qatext=np.delete(self.qatext,delindexes)
#         print(self.qatext[:3])  
#         print(self.train_data[:3])
        self.val_data,self.val_label,self.val_data_text=self.load_test()
        
        #print(self.val_label)
        # normalize scores
        #print(self.val_data)
        ids=np.array([self.val_data[i][0] for i in range(self.val_data.shape[0])])
        #print(ids[:10])
        norm_lbls=np.zeros(len(self.val_label))
        
        for i in set(ids): 
            #print(i)
            ind=np.where(ids==i)
            #print(ind)
            #print(self.val_label[ind])
            minscoe=min(self.val_label[ind])
            if minscoe<0:
                #print(self.train_label[ind])
                self.val_label[ind]=self.val_label[ind]+(-1.0*minscoe)
                #print(self.train_label[ind])
            maxscoe=max(self.val_label[ind])
            if maxscoe==0:
                self.val_label[ind]+=1
            #print(self.train_label[ind]) 
            
            norm_lbls[ind]=20*(self.val_label[ind]/np.sum(self.val_label[ind]))+5
            #print(norm_lbls[ind])
            #print(20*(self.val_label[ind]/np.sum(self.val_label[ind]))+5)
        #print(norm_lbls[:10])
        norm_lbls=np.array(norm_lbls)
        #ind=np.where(self.val_label!=0)
        #print(self.val_label[:20])
        #self.val_label[ind]=norm_lbls[ind]
        self.val_label=norm_lbls
        #print(self.val_label[:20])
        #print(self.val_data)
        
        
    def load_test(self):        
        
        INPUT=self.dataset+self.parsed+"/test_data.txt"        
        fin_test=open(INPUT)        
        test=fin_test.readline().strip()
        test_data=[]
        
        while test:
            data=test.split(";")
            lst=[]
            for d in data[0].split(" "):
                lst.append(int(d)) 
            
            alst=[]
            
            for d in data[1].split(" ")[0::3]:
                alst.append(int(d))
            
            anlst=[]
            for d in data[1].split(" ")[1::3]:
                anlst.append(int(d))
            scoresanlst=[]
            for d in data[1].split(" ")[2::3]:
                scoresanlst.append(int(d))
                
            test_data.append([lst,alst,anlst,scoresanlst])
            
            test=fin_test.readline().strip()
        fin_test.close()       
        INPUT=self.dataset+self.parsed+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])     
        user_id_map={}
        INPUT3=self.dataset+self.parsed+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=uname.strip()
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
        fin.close()    
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        INPUT=self.dataset+self.parsed+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=self.dataset+self.parsed+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=self.dataset+self.parsed+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=self.dataset+self.parsed+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:] 
        
        Q_id_map_to_original={}
        INPUT2=self.dataset+self.parsed+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map_to_original[int(e[0])]=int(e[1])
            
        max_q_len=20
        max_d_len=100
        u_answers={}
        INPUT=self.dataset+self.parsed+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[int(d[0])]=d[1::2]
                
        
        batch_size=1     
        #results=[]        
        iii=0
        val_data=[]
        val_labels=[]
        val_qatext=[]
        for tq in test_data:
            #print(iii)
            iii=iii+1
            #print("test q:")
            #print(tq)            
           
            ids=tq[1] 
            answerids=tq[2]
            scoresanlst=tq[3]
            #print("experts:")      
            #print(ids)
            inputs=[]
            inputtext=[]
            
            qtext=[]
            qid=Q_id_map_to_original[int(tq[0][0])]
            qtext1=qtitle[qid].copy()
            qtext1.extend(qcontent[qid])
            qtext1=qtext1[:20]
            qtext=qtext1.copy()
            #print(qtext)
            for i in range(len(qtext)):
                qtext[i]=vocab.index(qtext[i])+1
            
            #if len(qtext)<max_q_len:                
            #        for i in range(max_q_len-len(qtext)):
                        #qtext.append(0)
            kkk=0
            for e in ids:              
                answerid=answerids[kkk]
                
                etext1=[]
                if answerid!=-1:
                    etext1=answers[int(answerid)][:100]
                    etext=etext1
                else:       
                    for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                    etext=etext1
                    #etext=etext1
                    if len(etext1)>max_d_len:                         
                            etext=random.sample(etext1,max_d_len)
                        
                
                #print(etext)
                
                for ii in range(len(etext)):
                    etext[ii]=vocab.index(etext[ii])+1
                
                #if len(etext)<max_d_len:                
                    #for i in range(max_d_len-len(etext)):
                        #etext.append(0)
                
                testlst=tq[0][0:2]
                testlst.append(user_id_map[str(e)])
                testlst=np.concatenate((testlst,[answerid],tq[0][2:]))        
                inputs.append(testlst)
                inputtext.append([qtext,etext]) 
                
                val_data.append(testlst)
                val_labels.append(float(scoresanlst[kkk]))
                val_qatext.append([qtext,etext])
                kkk+=1
                
            
        return np.array(val_data), np.array(val_labels), np.array(val_qatext)   
    
    def loadG(self):
        #load graph 
        INPUT=self.dataset+self.parsed+"/CQG.txt"
        self.G=nx.Graph();        
        self.G=nx.read_weighted_edgelist(INPUT)
        
        #get list of nodes
        order = [str(i) for i in range(len(self.G.nodes()))]
        
        #construct the adjacency matrix
        A =to_numpy_matrix(self.G, nodelist=order,dtype=np.float32)
        
        #construct the  identity matrix
        I = np.eye(self.G.number_of_nodes(),dtype=np.float32)    

        # add self-loop to the adjacency matrix
        A_tilde = A + I        
        print("A_tilde=",A_tilde.shape)
        
        # noemalize A_tilde using the degree matrix to obtain A_hat matrix
        Dhat05 = np.array(np.sum(A_tilde, axis=0),dtype=np.float32)[0]
        Dhat05 = np.matrix(np.diag(Dhat05),dtype=np.float32)  
        Dhat05=fractional_matrix_power(Dhat05,-0.5)  
        self.A_hat=np.array(Dhat05*A_tilde*Dhat05,dtype=np.float32)
        print("A_hat=",self.A_hat.shape)
    
    #copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
    @staticmethod
    def kernal_mus(n_kernels, use_exact):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu
     
    #copied from knrm paper copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
    @staticmethod
    def kernel_sigmas(n_kernels, lamb, use_exact):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma
    
    def q_a_rbf(self,inputs_q,inputs_d):    
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        
        
        self.max_q_len=len(inputs_q[0])
        self.max_d_len=len(inputs_d[0])
        
        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')
        batch_size=1
        
        # normalize and compute similarity matrix using l2 norm         
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2))
        #print(norm_q)
        norm_q=tf.reshape(norm_q,(len(norm_q),len(norm_q[0]),1))
        #print(norm_q)
        normalized_q_embed = q_embed / norm_q
        #print(normalized_q_embed)
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2))
        norm_d=tf.reshape(norm_d,(len(norm_d),len(norm_d[0]),1))
        normalized_d_embed = d_embed / norm_d
        #print(normalized_d_embed)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])
        #print(tmp)
        sim =tf.matmul(normalized_q_embed, tmp)
        #print(sim)        
        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [batch_size, self.max_q_len, self.max_d_len, 1])
        #print(rs_sim)
        
        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, self.mus)) / (tf.multiply(tf.square(self.sigmas), 2)))
        #print(tmp)
        
        feats = []  # store the soft-TF features from each field.
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]
        
        aggregated_kde = tf.reduce_sum(kde , [1])  # [batch, n_bins]   *q_weights
        #print( aggregated_kde)
        feats.append(aggregated_kde) # [[batch, nbins]]
        feats_tmp = tf.concat( feats,1)  # [batch, n_bins]
        #print ("batch feature shape:", feats_tmp.get_shape())
        
        # Reshape. (maybe not necessary...)
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        feats_flat2=tf.reshape(feats_flat, [1,self.n_bins])
        
        return(feats_flat2)               
        
    def GCN_layer1(self):      
        return tf.matmul(self.A_hat,self.GCNW_1)
    
    def GCN_layer2(self,i,X):
        a=tf.matmul([self.A_hat[i,:]],X) 
        return tf.matmul(a,self.GCNW_2)
    
    def GCN_layers(self,i):
        H_1 = self.GCN_layer1()        
        H_2 = self.GCN_layer2(i,H_1)
        return H_2
              
    def model_test(self):
        embed=[]
        #print(self.inputs)
        for k in range(len(self.inputs)): 
            ind=self.inputs[k]
            qtext=[self.qatextinput[k][0]]
            atext=[self.qatextinput[k][1]]
            #print(qtext)
            #print(atext)
            q_a_rbf=self.q_a_rbf(qtext,atext)
            #print(q_a_rbf)
            qembed=tf.constant([np.zeros(self.d)],dtype=tf.float32)
            #print(qembed)
            askerembed=self.GCN_layers(ind[1])
            answererembed=self.GCN_layers(ind[2])
            i=1
            lst=[self.GCN_layers(ind[3])]
            for indx in  ind[4:]:
                lst.append(self.GCN_layers(indx))
                i=i+1
            
            tagsembed=tf.math.reduce_sum(lst, axis=0)/i            
            embed1=tf.concat([q_a_rbf,qembed,askerembed,answererembed,tagsembed],1, name='concat')
            #embed1=tf.concat([qembed,askerembed,answererembed,tagsembed],1, name='concat')
            embed.append(embed1)
        embed=tf.reshape(embed,[len(self.inputs),self.regindim])    
        #return  tf.reshape(tf.matmul(embed,self.W4),[len(self.inputs)]) + self.b
        #print(embed)
        #print(len(embed))
        #print(len(embed[0]))
        w1out=tf.nn.tanh(tf.matmul(embed,self.W1))
        #print(w1out.shape)
        #w2out=tf.nn.tanh(tf.matmul(w1out,self.W2))
        #print(w2out.shape)
        #w3out=tf.nn.tanh(tf.matmul(w2out,self.W3))
        #print(w3out.shape)   
        return  tf.reshape(tf.matmul(w1out,self.W4),[len(self.inputs)]) + self.b
    
    def model(self):
        embed=[]
        #print(self.inputs)
        for k in range(len(self.inputs)): 
            ind=self.inputs[k]
            qtext=[self.qatextinput[k][0]]
            #print(qtext)
            atext=[self.qatextinput[k][1]]
            #print(atext)
            #print(qtext)
            #print(atext)
            #sys.exit(0)
            q_a_rbf=self.q_a_rbf(qtext,atext)
            #print(q_a_rbf)
            qembed=self.GCN_layers(ind[0])
            #print(qembed)
            askerembed=self.GCN_layers(ind[1])
            answererembed=self.GCN_layers(ind[2])
            i=1
            lst=[self.GCN_layers(ind[4])]
            for indx in  ind[5:]:
                lst.append(self.GCN_layers(indx))
                i=i+1
            
            tagsembed=tf.math.reduce_sum(lst, axis=0)/i            
            embed1=tf.concat([q_a_rbf,qembed,askerembed,answererembed,tagsembed],1, name='concat')
            #embed1=tf.concat([qembed,askerembed,answererembed,tagsembed],1, name='concat')
            embed.append(embed1)
        embed=tf.reshape(embed,[len(self.inputs),self.regindim])    
        #return  tf.reshape(tf.matmul(embed,self.W4),[len(self.inputs)]) + self.b
        #print(embed)
        #print(len(embed))
        #print(len(embed[0]))
        w1out=tf.nn.tanh(tf.matmul(embed,self.W1))
        #print(w1out.shape)
        #w2out=tf.nn.tanh(tf.matmul(w1out,self.W2))
        #print(w2out.shape)
        #w3out=tf.nn.tanh(tf.matmul(w2out,self.W3))
        #print(w3out.shape)   
        return  tf.reshape(tf.matmul(w1out,self.W4),[len(self.inputs)]) + self.b
        
    
    def loss(self):
        self.L= tf.reduce_mean(tf.square(self.model() - self.outputs))#+0.5*tf.nn.l2_loss(self.W4)\
                        #+0.5*tf.nn.l2_loss(self.b) #+0.5*tf.nn.l2_loss(self.GCNW_1)+0.5*tf.nn.l2_loss(self.GCNW_2)
        return self.L  
        
    def train(self): 
        self.load_traindata(20,100)
        self.init_model()
        
        print("train data loaded!!")  
        
        len_train_data=len(self.train_data)
        val_len=len(self.val_data)
        loss_=0
        epochs = range(self.epochs)
        self.batch_size=1
        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.compat.v1.train.exponential_decay(0.0005,
                                        global_step, 1200,
                                        0.95, staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=decayed_lr,epsilon=5e-6)#(decayed_lr,epsilon=5e-6)
        
        outdir_model=self.dataset+self.parsed+"/model"
        
        if not os.path.exists(outdir_model):
            print("{} data dir not found.\n"
              " Creating a folder for that."
              .format(outdir_model))
            os.mkdir(outdir_model)
        
        logfile=open(outdir_model+"/log.txt","w")
        t_loss=[]
        v_loss=[]
        eps=[]
        
        for epoch in epochs:
            ind_new=[i for i in range(len_train_data)]
            np.random.shuffle(ind_new)
            self.train_data=self.train_data[ind_new,]
            self.train_label=self.train_label[ind_new,]           
            self.qatext=self.qatext[ind_new,]  
            
            start=0
            end=0
            for i in range(math.ceil(len_train_data/self.batch_size)):
                if ((i+1)*self.batch_size)<len_train_data:                    
                    start=i*self.batch_size
                    end=(i+1)*self.batch_size
                else:                    
                    start=i*self.batch_size
                    end=len_train_data
                    
                self.inputs=self.train_data[start:end]
                self.outputs=self.train_label[start:end]
                self.qatextinput=self.qatext[start:end]
                #print(i)
                #print(self.inputs)
                #print(self.outputs)
                #print(self.model())
                #print(self.qatextinput)
                
                opt.minimize(self.loss, var_list=[self.GCNW_1,self.GCNW_2,self.W1,self.W4,self.b,self.embeddings])#,self.W2,self.W3
                
                #q_embed = tf.nn.embedding_lookup(self.embeddings, self.qatextinput[0][0], name='qemb')
                #print(self.qatextinput[0][1])
                #print(self.outputs)
                #d_embed = tf.nn.embedding_lookup(self.embeddings, self.qatextinput[0][1], name='demb')
                #print(self.embeddings[0,:10])
                
                loss_+=self.L 
                
                global_step.assign_add(1)
                opt._decayed_lr(tf.float32)
                
                #print(self.Loss)
                #sys.exit(0)
                if (i+1)%50==0:                    
                    rep=(epoch*math.ceil(len_train_data/self.batch_size))+((i+1))
                    txt='Epoch %2d: i  %2d  out of  %4d     loss=%2.5f' %(epoch, i*self.batch_size, len_train_data, loss_/(rep))
                    logfile.write(txt+"\n")
                    print(txt)    
            #opt._decayed_lr(tf.float32)
            #print(self.W4)
            #validate the results
            print("\n************\nValidation started....\n")
            val_loss=0
            
            for ii in range(math.ceil(val_len/self.batch_size)):
                if ((ii+1)*self.batch_size)<val_len:
                    start=ii*self.batch_size
                    end=(ii+1)*self.batch_size
                else:
                    start=ii*self.batch_size
                    end=val_len
                self.inputs=self.val_data[start:end]
                self.outputs=self.val_label[start:end]
                self.qatextinput=self.val_data_text[start:end]
                val_loss+=self.loss()
                #print(self.loss())
                #print(val_loss)
                if (ii+1)%50==0:                   
                    txt='Epoch %2d: ii  %2d  out of  %4d     validation loss=%2.5f' %(epoch, ii*self.batch_size, val_len, val_loss/(ii+1))
                    logfile.write(txt+"\n")
                    print(txt)
            txt='Epoch %2d: ii  %2d  out of  %4d     validation loss=%2.5f' %(epoch, ii*self.batch_size, val_len, val_loss/(ii+1))
            logfile.write(txt+"\n")
            print(txt)
            
            if epoch%1==0:
                pkl_filename =outdir_model+"/pickle_model.pkl"+str(epoch)
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(self, file)
                print("model was saved")
            t_loss.append(loss_/(rep))
            v_loss.append(val_loss/math.ceil(val_len/self.batch_size))
            eps.append(epoch)
            plt.figure(figsize=(10,7))
            plt.plot(eps,t_loss,'r-o',label = "train loss")
            plt.plot(eps,v_loss,'b-*',label = "validation loss")
            plt.title("train and validation losses")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            plt.savefig(outdir_model+ "/loss"+str(epoch)+".png")
            plt.show()
        print("train model done!!")
        logfile.close() 
        #print(self.W4)
        plt.figure(figsize=(10,7))
        plt.plot(eps,t_loss,'r-o',label = "train loss")
        plt.plot(eps,v_loss,'b-*',label = "validation loss")
        plt.title("train and validation losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig(outdir_model+ "/loss.png")
        plt.show()
    
        
    def saveembedings(self):
        path=self.dataset+self.parsed+"/model/"
        embedding_file=open(path+"G_embeddings.txt","w")
        for i in range(self.N):
              emb_i=self.GCN_layers(i).numpy()[0]
              embedding_file.write(str(i)+": "+str(emb_i)+"\n")
        embedding_file.close()
        embedding_file=open(path+"word_embeddings.txt","w")         
        for i in range(1,self.vocab_size+1):
              emb_i=self.embeddings[i].numpy()
              embedding_file.write(str(i)+": "+str(emb_i)+"\n")
        embedding_file.close() 
        print("Embeddings were stored at: "+path)
        
    def test_model_allanswerers(dataset,modelname,path):        
        pkl_filename =dataset+path+ "/model/"+modelname
        # Load from file
        with open(pkl_filename, 'rb') as file:
            ob = pickle.load(file)
        print("model was loaded!!")
        #print(regr.get_params(deep=True))        
        #print(tf.reshape(ob.W4,(4,32)))
        #sys.exit(0)
        
        INPUT=dataset+path+"/"+"test_data.txt"
        
        fin_test=open(INPUT)        
        test=fin_test.readline().strip()
        test_data=[]
        
        while test:
            data=test.split(";")
            lst=[]
            for d in data[0].split(" "):
                lst.append(int(d)) 
            
            alst=[]
            
            for d in data[1].split(" ")[0::3]:
                alst.append(int(d))
            
            anlst=[]
            for d in data[1].split(" ")[1::3]:
                anlst.append(int(d))
                
            test_data.append([lst,alst,anlst])
            
            test=fin_test.readline().strip()
        fin_test.close()       
        
        INPUT=dataset+path+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])
        
                
        user_id_map={}
        INPUT3=dataset+path+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=uname.strip()
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
            
        fin.close()
        
        
        
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        
        INPUT=dataset+path+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=dataset+path+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=dataset+path+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=dataset+path+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:] 
        
        Q_id_map_to_original={}
        INPUT2=dataset+path+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map_to_original[int(e[0])]=int(e[1])
            
        
        allanswererids=[]
        INPUT=dataset+path+"/user_tags.txt"
        fin=open(INPUT,"r")
        line=fin.readline()#skip file header
        line=fin.readline().strip()#read first line
        while line:
            allanswererids.append(int(line.split(" ")[0]))
            line=fin.readline().strip()
        fin.close()
        allanswererids=np.array(allanswererids)
        
        max_q_len=ob.max_q_len
        ob.max_d_len=1*ob.max_d_len
        max_d_len=ob.max_d_len
        max_q_len=20
        max_d_len=100
        u_answers={}
        INPUT=dataset+path+"/user_answers.txt"
        
        
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[int(d[0])]=d[1::2]
                
                
        
        batch_size=1        
        OUTPUT=dataset+path+"/model/test_results_all_"+modelname+".txt"
        fout=open(OUTPUT,"w")
        #results=[]        
        iii=0
        for tq in test_data:
            print(iii)
            iii=iii+1
            print("test q:")
            print(tq)
            
            alleids=list(np.setdiff1d(allanswererids,tq[1]))
            allaids=[-1]*len(alleids)
            
            ids=tq[1]
            ids.extend(alleids)
            answerids=tq[2]
            answerids.extend(allaids)
            
            print("experts:")      
            print(ids)
            inputs=[]
            inputtext=[]
            
            qtext=[]
            qid=Q_id_map_to_original[int(tq[0][0])]
            qtext1=qtitle[qid].copy()
            qtext1.extend(qcontent[qid])
            qtext1=qtext1[:20]
            qtext=qtext1.copy()
            #print(qtext)
            for i in range(len(qtext)):
                qtext[i]=vocab.index(qtext[i])+1
            
            #if len(qtext)<max_q_len:                
            #        for i in range(max_q_len-len(qtext)):
                        #qtext.append(0)
            kkk=0
            for e in ids:              
                answerid=answerids[kkk]
                kkk+=1
                etext1=[]
                if answerid!=-1:
                    etext1=answers[int(answerid)][:100]
                    etext=etext1
                else:       
                    for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                    etext=etext1
                    #etext=etext1
                    if len(etext1)>max_d_len:                         
                            etext=random.sample(etext1,max_d_len)
                        
                
                #print(etext)
                
                for ii in range(len(etext)):
                    etext[ii]=vocab.index(etext[ii])+1
                
                #if len(etext)<max_d_len:                
                    #for i in range(max_d_len-len(etext)):
                        #etext.append(0)
                
                testlst=tq[0][0:2]
                testlst.append(user_id_map[str(e)])
                testlst=np.concatenate((testlst,tq[0][2:]))        
                inputs.append(testlst)
                inputtext.append([qtext,etext])           
            
            ob.inputs=inputs
            ob.qatextinput=inputtext
            #print(ob.inputs)
            #print(inputtext[0:2])
            s=ob.model_test().numpy() 
            print(s)
            res=""
            for i in range(len(ids)):
                res+=str(ids[i])+" "+ str(s[i])+";" 
            
            #res=" ".join([str(r) for r in sorted_ids[0:topk]])
            fout.write(res.strip()+"\n")
            fout.flush()
        fout.close()
        #OUTPUT=dataset+"/ColdEndFormat/EndCold_test_results.txt" 
        #np.savetxt(OUTPUT,np.array(results), fmt='%d')
        print("test_model done!!") 

    def prepare_train_test(data):
       ob=AttHetRL(data)
       ob.prepare_train_test_data()
       
    def run_train(data,dim_node=32,dim_word=300,epochs=10,batch_size=16): 
        ob=AttHetRL(data,dim_node,dim_word,epochs,batch_size)
        ob.train()
        ob.saveembedings()
        


#dataset=["android"] 
#data="../data/"+dataset[0]

#step 1
#trian=True
#if trian==True:
#    ob=AttHet(data)    
    #ob.prepare_train_test_data()    
#    ob.train() 
#else:    
#    AttHet.test_model(data,"pickle_QR_model.pkl2_ng","/parsed")
    #AttHet..test_model_allanswerers(data,"pickle_QR_model.pkl2_ng","/parsed")
#print("Done!")
        