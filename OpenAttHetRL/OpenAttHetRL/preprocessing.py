import os
try:
    import ujson as json
except:
    import json
import sys
from lxml import etree
from bs4 import BeautifulSoup
import string, random
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.data.path.append("../data/nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import logging
from collections import Counter
import numpy as np
import re
from shutil import copyfile

class datapreprocessing:     
    def __init__(self,dataset,path):
        self.data_name=dataset
        self.path=path
        self.part_user = set()        
        self.count_Q, self.count_A = {}, {}
        self.qa_map = {}
        self.test_candidates = set()
        self.parsed="/parsed"
        
    
    def clean_html(self,x):
        return BeautifulSoup(x, 'lxml').get_text()  
    
    
    def clean_str(self,s):
        """Clean up the string

        * New version, removing all punctuations

        Cleaning strings of content or title
        Original taken from [https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py]

        Args:
            string - the string to clean

        Return:
            _ - the cleaned string
        """
        ss = s
        translator = str.maketrans("", "", string.punctuation)
        ss = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", ss)
        ss = re.sub(r"\'s", "s", ss)
        ss = re.sub(r"\'ve", "ve", ss)
        ss = re.sub(r"n\'t", "nt", ss)
        ss = re.sub(r"\'re", "re", ss)
        ss = re.sub(r"\'d", "d", ss)
        ss = re.sub(r"\'ll", "ll", ss)
        ss = re.sub(r"\s{2,}", " ", ss)
        ss = ss.translate(translator)
        ss =' '.join([item for item in ss.split(" ") if (not item.isdigit()) and len(item.strip())>1])
        return ss.strip().lower()


    def remove_stopwords(self,string, stopword_set):
        """Removing Stopwords

        Args:
            string - the input string to remove stopwords
            stopword_set - the set of stopwords

        Return:
            _ - the string that has all the stopwords removed
        """        
        word_tokens = word_tokenize(string)        
        filtered_string = [word for word in word_tokens
                           if word not in stopword_set]
        return " ".join(filtered_string)
    
    def load_data1(self):
        qinfof=open(self.path+self.data_name+"/questionsinfo.txt") # file format: qnewid qoriginalid qtags note that in our data files qnewid starts from 1
        #qidmap map the question ids into the original ides. 
        #keys are the ides used in our data and the value is the original ids        
        self.qidmap=[]
        line=qinfof.readline() #skip header
        line=qinfof.readline().strip() # read first line 
        while line:
            ids=line.split(" ")
            self.qidmap.append(int(ids[1]))
            line=qinfof.readline().strip()
        #print(self.qidmap)
        qinfof.close()
        
        #load q info from post.xml      
        parser = etree.iterparse(self.path+self.data_name+"/Posts.xml", events=('end',), tag='row')
        fout_q= open(self.path+self.data_name+ self.parsed+"/Posts_Q.json", "w") 
        fout_a= open(self.path+self.data_name+self.parsed+"/Posts_A.json", "w") 
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Body'] = self.clean_html(attr['Body'])
            if attr['PostTypeId'] == '1' and int(attr['Id']) in self.qidmap:
                fout_q.write(json.dumps(attr) + "\n")
            elif attr['PostTypeId'] == '2' and int(attr['ParentId']) in self.qidmap and int(attr['Score'])> 0:
                fout_a.write(json.dumps(attr) + "\n")
            
        fout_q.close()
        fout_a.close()
        print("done!!!")   
    
    def load_data(self,minscore):       
        
        #load q info from post.xml      
        parser = etree.iterparse(self.path+self.data_name+"/Posts.xml", events=('end',), tag='row')
        fout_q= open(self.path+self.data_name+self.parsed+ "/Posts_Q.json", "w") 
        fout_a= open(self.path+self.data_name+ self.parsed+"/Posts_A.json", "w") 
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Body'] = self.clean_html(attr['Body'])
            if attr['PostTypeId'] == '1':
                fout_q.write(json.dumps(attr) + "\n")
            elif attr['PostTypeId'] == '2' and int(attr['Score'])>= minscore:
                fout_a.write(json.dumps(attr) + "\n")
            
        fout_q.close()
        fout_a.close()
        print("done!!!")
    
    def process_QA(self,min_a_size):
        """Process QA

        Extract attributes used in this project
        Get rid of the text information,
        only record the question-user - answer-user relation

        Args:
            data_dir - the dir where primitive data is stored
        """
        data_dir=self.path+self.data_name+self.parsed+"/"
        POST_Q = "Posts_Q.json"
        POST_A = "Posts_A.json"
        OUTPUT = "Record_All.json"
        RAW_STATS = "question.stats.raw"

        # Get logger to log exceptions
        logger = logging.getLogger(__name__)

        no_acc_question = 0

        raw_question_stats = []

        if not os.path.exists(data_dir + POST_Q):
            raise IOError("file {} does NOT exist".format(data_dir + POST_Q))

        if not os.path.exists(data_dir + POST_A):
            raise IOError("file {} does NOT exist".format(data_dir + POST_A))

        # Process question information
        with open(data_dir + POST_Q, 'r') as fin_q:
            for line in fin_q:
                data = json.loads(line)
                try:
                    qid, rid = data.get('Id', None), data.get('OwnerUserId', None)
                    # If such 
                    if qid and rid:
                        acc_id = data.get('AcceptedAnswerId', None)
                        answer_count = int(data.get('AnswerCount', -1))
                        tags=data.get('Tags').strip()
                        tagsarr=[]
                        if len(tags)>0:
                            tagsarr=tags[1:].replace(">","").split("<")
                        if acc_id:
                            self.qa_map[qid] = {
                                'QuestionId': qid,
                                'QuestionOwnerId': rid,
                                'AcceptedAnswerId': acc_id,
                                'AcceptedAnswererId': None,
                                'AnswererIdList': [],
                                'AnswererAnswerTuples': [],
                                'Tags':tagsarr
                            }
                            self.count_Q[rid] = self.count_Q.get(rid, 0) + 1
                        else:
                            no_acc_question += 1

                        if answer_count >= 0:
                            raw_question_stats.append(answer_count)
                except:
                    logger.error("Error at process_QA 1: " + str(data))
                    continue
        print("\t\t{} questions do not have accepted answer!"
              .format(no_acc_question))

        # Count raw question statistics
        raw_question_stats_cntr = Counter(raw_question_stats)
        with open(data_dir + RAW_STATS, "w") as fout:
            for x in sorted(list(raw_question_stats_cntr.keys())):
                print("{}\t{}".format(x, raw_question_stats_cntr[x]), file=fout)
            print("Total\t{}".format(sum(raw_question_stats)), file=fout)

        # Process answer information
        with open(data_dir + POST_A, 'r') as fin_a:
            for line in fin_a:
                data = json.loads(line)
                try:
                    answer_id = data.get('Id', None)
                    aid = data.get('OwnerUserId', None)
                    qid = data.get('ParentId', None)
                    entry = self.qa_map.get(qid, None)
                    if answer_id and aid and qid and entry:
                        entry['AnswererAnswerTuples'].append((aid, answer_id))
                        entry['AnswererIdList'].append(aid)
                        self.count_A[aid] = self.count_A.get(aid, 0) + 1

                        # Check if we happen to hit the accepted answer
                        if answer_id == entry['AcceptedAnswerId']:
                            entry['AcceptedAnswererId'] = aid
                    else:
                        logger.error(
                            "Answer {} belongs to unknown Question {} at Process QA"
                            .format(answer_id, qid))
                except IndexError as e:
                    logger.error(e)
                    logger.info("Error at process_QA 2: " + str(data))
                    continue

        # Fill in the blanks of `AcceptedAnswererId`
        # for qid in self.qa_map.keys():
        #    acc_id = self.qa_map[qid]['AcceptedAnswerId']
        #    for aid, answer_id in self.qa_map[qid]['AnswererAnswerTuples']:
        #        if answer_id == acc_id:
        #            self.qa_map[qid]['AcceptedAnswererId'] = aid
        #            break
        # remove qid if qid doesn't have an answer or accepted anser
        qid_list = list(self.qa_map.keys())
        for qid in qid_list:
            if len(self.qa_map[qid]['AnswererIdList'])<min_a_size\
                or not self.qa_map[qid]['AcceptedAnswererId']:
                del self.qa_map[qid]
                
        print("\t\tWriting the Record for ALL to disk.")
        with open(data_dir + OUTPUT, 'w') as fout:
            for q in self.qa_map.keys():
                fout.write(json.dumps(self.qa_map[q]) + "\n")
        
        print("QA processing done!!")
    
    def question_stats(self):
        """Find the question statistics for `Introduction`

        Args:
            data_dir -
        Return
        """
        data_dir=self.path+self.data_name+self.parsed+"/"
        OUTPUT = "question.stats"
        count = []
        for qid in self.qa_map.keys():
            ans_count = len(self.qa_map[qid]['AnswererIdList'])
            count.append(ans_count)
            if ans_count == 0:
                print("0 answer id list", qid)
        question_stats_cntr = Counter(count)

        with open(data_dir + OUTPUT, "w") as fout:
            for x in sorted(list(question_stats_cntr.keys())):
                print("{}\t{}".format(x, question_stats_cntr[x]), file=fout)
            print("Total\t{}".format(sum(count), file=fout), file=fout)
        
        print("question_stats done!!!")

    def extract_question_content(self):
        """Extract questions, content pairs from question file

        Question content pair format:
            <qid> <content>
        We extract both with and without stop-word version
            which is signified by "_nsw"

        Args:
            data_dir - data directory
            parsed_dir - parsed file directory
        """
        data_dir=parsed_dir=self.path+self.data_name+self.parsed+"/"
        INPUT = "Posts_Q.json"
        OUTPUT_T = "Q_title.txt"  # Question title
        OUTPUT_T_NSW = "Q_title_nsw.txt"  # Question title, no stop word
        OUTPUT_C = "Q_content.txt"  # Question content
        OUTPUT_C_NSW = "Q_content_nsw.txt"  # Question content, no stop word

        logger = logging.getLogger(__name__)

        if not os.path.exists(data_dir + INPUT):
            IOError("Can NOT locate {}".format(data_dir + INPUT))
      
        sw_set = set(stopwords.words('english'))  # Create the stop word set
        #print(sw_set)
        #qid_list = list(self.qa_map.keys())
        #print(len(qid_list))
        # We will try both with or without stopwords to
        # check out the performance.
        with open(data_dir + INPUT, "r") as fin, \
                open(parsed_dir + OUTPUT_T, "w") as fout_t, \
                open(parsed_dir + OUTPUT_T_NSW, "w") as fout_t_nsw, \
                open(parsed_dir + OUTPUT_C, "w") as fout_c, \
                open(parsed_dir + OUTPUT_C_NSW, "w") as fout_c_nsw:
            for line in fin:
                data = json.loads(line)                
                try:
                    qid = data.get('Id')
                    #print("qid:"+qid)                    
                    if qid not in self.qa_map:                        
                        #print("erorr qid not in self.qa_map")
                        continue                    
                    
                    title = data.get('Title')
                    content = data.get('Body')

                    content, title = self.clean_str(content), self.clean_str(title)                    
                    content_nsw = self.remove_stopwords(content, sw_set)                    
                    title_nsw = self.remove_stopwords(title, sw_set)                  
                    
                    print("{} {}".format(qid, content_nsw),
                          file=fout_c_nsw)  # Without stopword
                    print("{} {}".format(qid, content),
                          file=fout_c)  # With stopword
                    print("{} {}".format(qid, title_nsw),
                          file=fout_t_nsw)  # Without stopword
                    print("{} {}".format(qid, title),
                          file=fout_t)  # With stopword
                except:
                    logger.info("Error at Extracting question content and title: "
                                + str(data))
                    continue
        print("extract_question_content done!!!")   
    
    
    def extract_answer_content(self):
        """Extract answers, content pairs from Post_A.json file

        Answer content pair format:
            <answerid> <content>
        We extract both with and without stop-word version
            which is signified by "_nsw"
        """
        data_dir=self.path+self.data_name+self.parsed+"/"
        INPUT = "Posts_A.json"        
        OUTPUT_C = "A_content.txt"  # Question content
        OUTPUT_C_NSW = "A_content_nsw.txt"  # Question content, no stop word

        logger = logging.getLogger(__name__)

        if not os.path.exists(data_dir + INPUT):
            IOError("Cannot find {}".format(data_dir + INPUT))
      
        sw_set = set(stopwords.words('english'))  # Create the stop word set
        
        with open(data_dir + INPUT, "r") as fin, open(data_dir + OUTPUT_C, "w") as fout_c, \
                open(data_dir + OUTPUT_C_NSW, "w") as fout_c_nsw:
             for line in fin:
                data = json.loads(line)                
                try:
                    qid = data.get('ParentId')
                    #print("qid:"+qid)                    
                    if qid not in self.qa_map:                        
                        #print("erorr qid not in self.qa_map")
                        continue                   
                    
                    answerid=data.get('Id')
                    content = data.get('Body')

                    content= self.clean_str(content)                   
                    content_nsw = self.remove_stopwords(content, sw_set)                  
                    
                    print("{} {}".format(answerid, content_nsw),
                          file=fout_c_nsw)  # Without stopword
                    print("{} {}".format(answerid, content),
                          file=fout_c)  # With stopword
                except:
                    logger.info("Error at Extracting question content and title: "
                                + str(data))
                    continue
        print("extract_answer_content done!!!")
    
    
    def build_test_set(self,threshold, test_sample_size,
                   test_proportion):
        """
        Building test datase,
        test_proportiont
        Args:
            parse_dir - the directory to save parsed set.
            threshold - the selection threshold the q raiser and q accepted answerer should ask and answer at least threshold quetions
            test_sample_size: number of answerer for each test question. if the number of answerers are grather than this sample them
            otherwise add some negetive samples to have test_sample_size anserers for each quetion
        Return:
        """
        data_dir=parsed_dir=self.path+self.data_name+self.parsed+"/"
        
        TEST = "test.txt" 
        #format: contains test questions infos note that for each test Q there are n answerer ids. 
        #if in the quetion the number of answerers are grather than n sample n,otherwise add some negative samples to have n ids.
        #Format:
        #Qownerid Qid AcceptedAnswererId(orBestanswereid) AnswererId1 AnswererId2 ... AnswererIdn
        TEST_q_answer = "test_q_answer.txt" #format qid answerid1 ... answeridn
        
        OUTPUT_TRAIN = "Record_Train.json"

        accept_no_answerer = 0

        ordered_count_A = sorted(
            self.count_A.items(), key=lambda x:x[1], reverse=True)
        ordered_aid = [x[0] for x in ordered_count_A]
        ordered_aid = ordered_aid[: int(len(ordered_aid) * 0.1)]

        
                
        question_count = len(self.qa_map)

        for qid in self.qa_map.keys():
            accaid = self.qa_map[qid]['AcceptedAnswererId']
            rid = self.qa_map[qid]['QuestionOwnerId']
            if not accaid:
                accept_no_answerer += 1
                continue
            if self.count_Q[rid] >= threshold and self.count_A[accaid] >= threshold:
                self.test_candidates.add(qid)

        print("\t\tSample table size {}. Using {} instances for test."
              .format(len(self.test_candidates), int(question_count * test_proportion)))

        test = np.random.choice(list(self.test_candidates),
                                size=int(question_count * test_proportion),
                                replace=False)

        print("\t\tAccepted answer without Answerer {}".format(accept_no_answerer))

        print("\t\tWriting the sampled test set to disk")
        with open(parsed_dir + TEST, "w") as fout, open(parsed_dir +TEST_q_answer, "w") as fout_q_answer:
            for qid in test:
                rid = self.qa_map[qid]['QuestionOwnerId']
                accaid = self.qa_map[qid]['AcceptedAnswererId']
                aid_list = self.qa_map[qid]['AnswererIdList']
                if len(aid_list) <= test_sample_size:
                    neg_sample_size = test_sample_size - len(aid_list)
                    neg_samples = random.sample(ordered_aid, neg_sample_size)
                    samples = neg_samples + aid_list
                else:
                    samples = random.sample(aid_list, test_sample_size)
                    if accaid not in samples:
                        samples.pop()
                        samples.append(accaid)
                samples = " ".join(samples)
                print("{} {} {} {}".format(rid, qid, accaid, samples),
                      file=fout)
                a_answer_lst=self.qa_map[qid]['AnswererAnswerTuples']
                if a_answer_lst:                      
                    ids=" ".join([row[1] for row in a_answer_lst])
                    fout_q_answer.write(str(qid)+" "+ids+"\n")

        # if qid is a test instance or qid doesn't have an answer
        qid_list = list(self.qa_map.keys())
        for qid in qid_list:
            if qid in test:
                #or len(self.qa_map[qid]['AnswererIdList'])<2\
                #or not self.qa_map[qid]['AcceptedAnswererId']:
                del self.qa_map[qid]

        # Write QA pair to file
        print("\t\tWriting the Record for training to disk")
        with open(data_dir + OUTPUT_TRAIN, 'w') as fout:
            for q in self.qa_map.keys():
                fout.write(json.dumps(self.qa_map[q]) + "\n")
        print("build_test_set done!!!")
    
    def build_test_with_all_answeres(self):
        """this generate test file with all answerers to be ranked by NErank model
        and the results are used to compare with team2box
        Format:
        rid qid bestanswerid userid_1 .... userid_n """
        #load all users' id 
        data_dir=self.path+self.data_name+self.parsed+"/"
        alluserids=""
        INPUT="user_tags.txt"
        fin=open(data_dir+INPUT,"r")
        line=fin.readline()#skip file header
        line=fin.readline().strip()#read first line
        while line:
            alluserids+=line.split(" ")[0]+" "
            line=fin.readline().strip()
        fin.close()
        
        #build file
        data_dir=self.path+self.data_name+self.parsed+"/"        
        INPUT="test.txt"
        OUTPUT="test_with_allusers.txt"
        with open(data_dir+INPUT,"r") as fin, open(data_dir+OUTPUT,"w") as fout:
            for line in fin:
                ids=line.split(" ")[0:3]
                fout.write(ids[0]+" "+ids[1]+" "+ids[2]+" "+alluserids.strip()+"\n")
        
    def extract_question_user(self):
        """Extract Question User pairs and output to file.
        Extract "Q" and "R". Format:
            <Qid> <Rid>
        E.g.
            101 40
            145 351

        Args:
            data_dir - data directory
            parsed_dir - parsed file directory
        """
        data_dir=parsed_dir=self.path+self.data_name+self.parsed+"/"
        # INPUT = "Record_Train.json"
        INPUT = "Record_All.json"
        OUTPUT = "Q_R.txt"
        OUTPUT_answer = "Q_answer.txt"

        if not os.path.exists(data_dir + INPUT):
            IOError("Can NOT find {}".format(data_dir + INPUT))

        with open(data_dir + INPUT, "r") as fin:
            with open(parsed_dir + OUTPUT, "w") as fout, open(parsed_dir + OUTPUT_answer, "w") as fout_answer:
                for line in fin:
                    data = json.loads(line)
                    qid = data['QuestionId']
                    rid = data['QuestionOwnerId']
                    self.part_user.add(int(rid))  # Adding participated questioners
                    print("{} {}".format(str(qid), str(rid)), file=fout)
                    a_answer_lst=data['AnswererAnswerTuples']
                    if a_answer_lst:                      
                        ids=" ".join([row[1] for row in a_answer_lst])
                        fout_answer.write(str(qid)+" "+ids+"\n")
                            
        print("extract_question_user done!!!")
    
    def extract_question_answer_user(self):
        """Extract Question, Answer User pairs and output to file.

        (1) Extract "Q" - "A"
            The list of AnswerOwnerList contains <aid>-<owner_id> pairs
            Format:
                <Qid> <Aid>
            E.g.
                100 1011
                21 490

        (2) Extract "Q" - Accepted answerer
            Format:
                <Qid> <Acc_Aid>
        Args:
            data_dir - data directory
            parsed_dir - parsed file directory
        """
        data_dir=parsed_dir=self.path+self.data_name+self.parsed+"/"
        INPUT = "Record_Train.json"
        OUTPUT_A = "Q_A.txt"
        OUTPUT_ACC = "Q_ACC.txt"

        if not os.path.exists(data_dir + INPUT):
            IOError("Can NOT find {}".format(data_dir + INPUT))

        with open(data_dir + INPUT, "r") as fin, \
                open(parsed_dir + OUTPUT_A, "w") as fout_a, \
                open(parsed_dir + OUTPUT_ACC, "w") as fout_acc:
            for line in fin:
                data = json.loads(line)
                qid = data['QuestionId']
                aid_list = data['AnswererIdList']
                accaid = data['AcceptedAnswererId']
                for aid in aid_list:
                    self.part_user.add(int(aid))
                    print("{} {}".format(str(qid), str(aid)), file=fout_a)
                print("{} {}".format(str(qid), str(accaid)), file=fout_acc)    
        print("extract_question_answer_user done!!!")
    
    def extract_answer_score(self):
        """Extract the answers vote, a.k.a. Scores.

        This information might be useful when
            the accepted answer is not selected.

        Args:
            data_dir - Input data dir
            parsed_dir - Output data dir
        """
        data_dir=parsed_dir=self.path+self.data_name+self.parsed+"/"
        INPUT = "Posts_A.json"
        OUTPUT = "A_score.txt"

        logger = logging.getLogger(__name__)

        if not os.path.exists(data_dir + INPUT):
            IOError("Cannot find file{}".format(data_dir + INPUT))

        with open(data_dir + INPUT, "r") as fin, \
            open(parsed_dir + OUTPUT, "w") as fout:
            for line in fin:
                data = json.loads(line)
                try:
                    qid = data.get('ParentId')
                    #print("qid:"+qid)                    
                    if qid not in self.qa_map:                        
                        #print("erorr qid not in self.qa_map")
                        continue 
                    aid = data.get('Id')
                    score = data.get('Score')
                    print("{} {}".format(aid, score), file=fout)
                except:
                    logging.info("Error at Extracting answer score: "
                                 + str(data))
                    continue
        print("extract_answer_score done!!!")

    
    
    def extract_question_best_answerer(self):
        """Extract the question-best-answerer relation

        Args:
            data_dir  - as usual
            parsed_dir  -  as usual
        """
        data_dir=parsed_dir=self.path+self.data_name+self.parsed+"/"
        INPUT_A = "Posts_A.json"
        # INPUT_MAP = "Record_Train.json"
        # Uncomment this when running NeRank
        INPUT_MAP = "Record_All.json"
        OUTPUT = "Q_ACC_A.txt"

        if not os.path.exists(data_dir + INPUT_A):
            IOError("Cannot find file {}".format(data_dir + INPUT_A))
        if not os.path.exists(data_dir + INPUT_MAP):
            IOError("Cannot find file {}".format(data_dir + INPUT_MAP))

        accanswerid_uaid = {}  # Accepted answer id to Answering user id
        answerid_score = {}  # Answer id to answer scores
        with open(data_dir + INPUT_MAP, "r") as fin_map, \
            open(parsed_dir + OUTPUT, "w") as fout:

            for line in fin_map:
                data = json.loads(line)
                try:
                    qid = data.get('QuestionId')
                    acc_aid = data.get("AcceptedAnswererId")
                    if qid and acc_aid:
                        print("{} {}".format(qid, acc_aid), file=fout)
                except:
                    print(1)
                    logging.info(
                        "Error at Extracting question, best answer user: "
                         + str(data))
        print("extract_question_best_answerer_2 done!!!")
    
    def write_part_users(self):
        parsed_dir=self.path+self.data_name+self.parsed+"/"
        OUTPUT = "QA_ID.txt"
        with open(parsed_dir + OUTPUT, "w") as fout:
            IdList = list(self.part_user)
            IdList.sort()
            for index, user_id in enumerate(IdList):
                print("{} {}".format(index + 1, user_id), file=fout)
        print("write_self.part_users done!!!")
          
    def ExtractText(self, maxqlen,maxalen,valpro):
        """
        convert the NeRank format data into team2box format
        args:
           maxlen: the maximum number of words in each question
           maxalen: the maximum number of words in each answer
        """
        #load answer score from A_score.txt
        data_dir=self.path+self.data_name+self.parsed+"/"
        outdir=self.path+self.data_name+self.parsed+"/"
        if not os.path.exists(outdir):
            print("{} data dir not found.\n"
              " Creating a folder for that."
              .format(outdir))
            os.mkdir(outdir)
        INPUT="A_score.txt"
        self.ascore_map={}
        with open(data_dir + INPUT, "r") as fin:
            for line in fin:
                aid,score=line.strip().split(" ")
                self.ascore_map[aid]=score
        
        #load answer content from A_content_nsw.txt        
        INPUT="A_content_nsw.txt"
        self.acontent_map={}  
        self.answerid_map=[]
        with open(data_dir + INPUT, "r") as fin:
            for line in fin:
                content=line.strip().split(" ")
                self.acontent_map[content[0]]=' '.join(content[1:])
                self.answerid_map.append(content[0].strip())
                
        #load question content from Q_content_nsw.txt        
        INPUT="Q_content_nsw.txt"
        self.qcontent_map={}       
        with open(data_dir + INPUT, "r") as fin:
            for line in fin:
                content=line.strip().split(" ")
                self.qcontent_map[content[0]]=' '.join(content[1:]) 
                
        #load question tritle from Q_title_nsw.txt        
        INPUT="Q_title_nsw.txt"
        self.qtitle_map={}
        with open(data_dir + INPUT, "r") as fin:
            for line in fin:
                content=line.strip().split(" ")
                self.qtitle_map[content[0]]=' '.join(content[1:])        
        
        OUTPUT_trian="train_text.txt"
        OUTPUT_trainqa="q_answer_ids_score.txt"
        INPUT="Record_Train.json"
        fout_trainqa=open(outdir + OUTPUT_trainqa, "w")
        fout_trainqa.write("Q_ID Answer_ID Score .....\n")
        answer_num=len(self.acontent_map)
        self.vcab={}
        train_count=test_count=val_count=0
        with open(data_dir + INPUT, "r") as fin,\
              open(outdir + OUTPUT_trian, "w") as fout_train :
            for line in fin:
                train_count+=1
                data = json.loads(line)
                qid=data.get('QuestionId')
                lstaid=[row[1] for row in data.get('AnswererAnswerTuples')]
                
                #get question words title+body
                qtitle=self.qtitle_map[qid].split(" ")
                qcontent=self.qcontent_map[qid].split(" ")
                cont=""
                if len(qtitle)<maxqlen:
                    cont=" ".join(qtitle).strip()
                    cont=cont+" "+" ".join(qcontent[:maxqlen-len(qtitle)]).strip()
                else:
                    cont= " ".join(qtitle[:maxqlen]).strip()                   
                
                fout_train.write(cont)
                fout_trainqa.write(qid)
                
                for vcb in cont.split(" "):
                    if vcb not in self.vcab:
                        self.vcab[vcb]=1
                    else:
                        self.vcab[vcb]=self.vcab[vcb]+1
                
                #add answers' content
                for aid in lstaid:
                    acont=self.acontent_map[aid].split(" ")
                    if len(acont)>maxalen:
                        acont=acont[:maxalen]
                    score=self.ascore_map[aid]    
                    fout_train.write(","+" ".join(acont).strip()+" "+score)   
                    fout_trainqa.write(" "+aid+" "+score)
                    for vcb in acont:
                        if vcb not in self.vcab:
                            self.vcab[vcb]=1
                        else:
                            self.vcab[vcb]=self.vcab[vcb]+1
                
                # add len(lstaid) negative samples into the treaing                 
                for i in range(len(lstaid)):
                    negaid=self.answerid_map[random.randint(0,answer_num-1)]
                    
                    while negaid in lstaid:
                        negaid=self.answerid_map[random.randint(0,answer_num-1)] 
                    
                    acont=self.acontent_map[negaid].split(" ")
                    if len(acont)>maxalen:
                        acont=acont[:maxalen]
                    fout_train.write(","+" ".join(acont).strip()+" 0")
                    for vcb in acont:
                        if vcb not in self.vcab:
                            self.vcab[vcb]=1
                        else:
                            self.vcab[vcb]=self.vcab[vcb]+1
                fout_train.write("\n")
                fout_trainqa.write("\n")
        
        OUTPUT_test="test_text.txt"        
        INPUT="test_q_answer.txt"        
        
        OUTPUT_val="validation_text.txt"        
        
        with open(data_dir + INPUT, "r") as fin,\
              open(outdir + OUTPUT_test, "w") as fout_test,\
              open(outdir + OUTPUT_val, "w") as fout_val:
            for line in fin:
                test_count+=1
                data = line.strip().split(" ")
                qid=data[0]
                lstaid=data[1:]
                
                #get question words title+body
                qtitle=self.qtitle_map[qid].split(" ")
                qcontent=self.qcontent_map[qid].split(" ")
                cont=""
                if len(qtitle)<maxqlen:
                    cont=" ".join(qtitle).strip()
                    cont=cont+" "+" ".join(qcontent[:maxqlen-len(qtitle)]).strip()
                else:
                    cont= " ".join(qtitle[:maxqlen]).strip()                    
                fout_test.write(cont)
                for vcb in cont.split(" "):
                        if vcb not in self.vcab:
                            self.vcab[vcb]=1
                        else:
                            self.vcab[vcb]=self.vcab[vcb]+1
                valcont=cont
                fout_trainqa.write(qid)
                #add answers' content
                for aid in lstaid:
                    acont=self.acontent_map[aid].split(" ")
                    if len(acont)>maxalen:
                        acont=acont[:maxalen]
                    score=self.ascore_map[aid]    
                    fout_test.write(","+" ".join(acont).strip()+" "+score)  
                    valcont=valcont+","+" ".join(acont).strip()+" "+score
                    fout_trainqa.write(" "+aid+" "+score)
                    for vcb in acont:
                        if vcb not in self.vcab:
                            self.vcab[vcb]=1
                        else:
                            self.vcab[vcb]=self.vcab[vcb]+1
                # add len(lstaid) negative samples into the treaing                 
                for i in range(len(lstaid)):
                    negaid=self.answerid_map[random.randint(0,answer_num-1)]
                    while negaid in lstaid:
                        negaid=self.answerid_map[random.randint(0,answer_num-1)] 
                    acont=self.acontent_map[negaid].split(" ")
                    if len(acont)>maxalen:
                        acont=acont[:maxalen]
                    fout_test.write(","+" ".join(acont).strip()+" 0")
                    valcont=valcont+ ","+" ".join(acont).strip()+" 0"
                    for vcb in acont:
                        if vcb not in self.vcab:
                            self.vcab[vcb]=1
                        else:
                            self.vcab[vcb]=self.vcab[vcb]+1
                fout_test.write("\n")
                fout_trainqa.write("\n") 
                if random.uniform(0, 1)<valpro:
                    val_count+=1
                    fout_val.write(valcont+"\n") 
        fout_trainqa.close()  
        
        OUTPUT="vocab.txt"
        outf=open(outdir + OUTPUT, "w")
        outf.write("vocab fequency\n")
        self.vcab = sorted(self.vcab.items(), key=lambda x: x[1], reverse=True)
        #print(self.vcab)
        self.vcab_map=[]
        for vcb in self.vcab:
            outf.write(vcb[0]+" "+str(vcb[1])+"\n") 
            self.vcab_map.append(vcb[0])
        outf.close()
        
        OUTPUT="properties.txt"
        outf=open(outdir + OUTPUT, "w")
        outf.write("vocab size="+str(len(self.vcab_map))+" train="+str(train_count)
                   +" test="+str(test_count)+" validation="+str(val_count)
                    +" qmaxlen="+str(maxqlen)+" answermaxlen="+str(maxalen))
        outf.close()
        print("done!!")
    
        
    def extract_q_tags(self):
        """get tags of each question and write in Q_tags.txt"""
        data_dir=self.path+self.data_name+self.parsed+"/"
        outdir=self.path+self.data_name+self.parsed+"/"
        INPUT="Record_All.json"
        OUTPUT="Q_tags.txt"
        with open(data_dir + INPUT, "r") as fin,\
              open(outdir + OUTPUT, "w") as fout: 
            for line in fin:
                data = json.loads(line)
                qid=data.get('QuestionId')
                tags=data.get('Tags')
                fout.write(qid+" "+" ".join(tags).strip()+"\n")
    
    def extract_user_tags_score(self):
        """get tags of each user and write in user_tags.txt
        and map of answer id and user id in answer_user_ids.txt"""
        
        data_dir=self.path+self.data_name+self.parsed+"/"
        outdir=self.path+self.data_name+self.parsed+"/"        
        
        INPUT="A_score.txt"
        self.ascore_map={}
        with open(data_dir + INPUT, "r") as fin:
            for line in fin:
                aid,score=line.strip().split(" ")
                self.ascore_map[aid]=score
        
        
        INPUT="Record_All.json"
        OUTPUT="user_tags.txt" 
        OUTPUT_auids="answer_user_ids.txt"
        user_tags={}
        with open(data_dir + INPUT, "r") as fin,\
              open(outdir + OUTPUT, "w") as fout,\
              open(outdir + OUTPUT_auids, "w") as fout_auids: 
            fout.write("answererId, tag numberofanswers score,....\n")
            fout_auids.write("answerID answererID\n")
            for line in fin:
                data = json.loads(line)
                user_answer_tuple=data.get('AnswererAnswerTuples')
                tags=data.get('Tags')
                for item in user_answer_tuple:
                    uid=item[0]
                    aid=item[1]
                    fout_auids.write(aid+" "+uid+"\n")
                    score=int(self.ascore_map[aid])
                    if uid not in user_tags:
                        user_tags[uid]={} 
                    for tag in tags:
                        if tag not in user_tags[uid]:
                            user_tags[uid][tag]=[1,score]
                        else: 
                            user_tags[uid][tag][0]+=1
                            user_tags[uid][tag][1]+=score
                
            for u in user_tags.keys():
                fout.write(u)
                for tag in user_tags[u].keys():
                    fout.write(" "+tag+" "+str(user_tags[u][tag][0])+" "+str(user_tags[u][tag][1]))   
                fout.write("\n")
                
    
    
    def createGraph(self):
        outputdirectory=self.parsed+"/"
        outdir=self.path+self.data_name+outputdirectory
        
        if not os.path.exists(outdir):
            print("{} data dir not found.\n"
              " Creating a folder for that."
              .format(outdir))
            os.mkdir(outdir)
        
        #load q tags        
        INPUT=self.path+self.data_name+self.parsed+"/Q_tags.txt"
        self.Q_id_map=[]
        self.Q_tags={}
        self.tag_id=[]
        
        with open( INPUT, "r") as fin:                
            for line in fin:
                data = line.strip().split(" ")
                qid=int(data[0])
                if qid not in self.Q_id_map:
                    self.Q_id_map.append(qid)
                
                qnewid=self.Q_id_map.index(qid)
                self.Q_tags[qnewid]=[]
                for tag in data[1:]:
                    if tag not in self.tag_id:
                        self.tag_id.append(tag)
                    tagid=self.tag_id.index(tag)
                    self.Q_tags[qnewid].append(tagid)    
                        
        
        INPUT=self.path+self.data_name+self.parsed+"/Record_All.json"
        #self.asker_id_map=[]
        #self.answerer_id_map=[]
        self.Q_asker={}
        self.Q_answerer={}
        self.user_id_map=[]  #contain both askers and answerers
        with open(INPUT, "r") as fin:
            for line in fin:
                data = json.loads(line)
                qid = int(data['QuestionId'])
                askerid=int(data['QuestionOwnerId'])
                
                #if askerid not in self.asker_id_map:
                #    self.asker_id_map.append(askerid) 
                
                if askerid not in self.user_id_map:
                    self.user_id_map.append(askerid)
                    
                askernewid=self.user_id_map.index(askerid)
                qnewid=self.Q_id_map.index(qid)
                self.Q_asker[qnewid]=askernewid
                
                answereridslst=data['AnswererIdList'] 
                self.Q_answerer[qnewid]=[]
                for answererid in answereridslst:
                    intaid=int(answererid)
                    if intaid not in self.user_id_map:
                        self.user_id_map.append(intaid)
                    intanewid=self.user_id_map.index(intaid)
                    if intanewid != self.Q_asker[qnewid]:
                        self.Q_answerer[qnewid].append(intanewid)
                    
                    
        numq=len(self.Q_id_map)
        #numasker=len(self.asker_id_map)
        numuser=len(self.user_id_map)
        numtags=len(self.tag_id)
        
        #write map ids into files
        OUTPUT=self.path+self.data_name+outputdirectory+"/Q_id_map.txt"
        fout=open(OUTPUT,"w")        
        for qid in self.Q_id_map:
            fout.write(str(self.Q_id_map.index(qid))+" "+str(qid)+"\n")
        fout.close()
        
        #OUTPUT=self.path+self.data_name+outputdirectory+"/asker_id_map.txt"
        #fout=open(OUTPUT,"w")        
        #for askerid in self.asker_id_map:
        #    fout.write(str(self.asker_id_map.index(askerid))+" "+str(askerid)+"\n")
        #fout.close()
        
        OUTPUT=self.path+self.data_name+outputdirectory+"/user_id_map.txt"
        fout=open(OUTPUT,"w")        
        for userid in self.user_id_map:
            fout.write(str(self.user_id_map.index(userid))+" "+str(userid)+"\n")
        fout.close()
        
        
        OUTPUT=self.path+self.data_name+outputdirectory+"/tag_id_map.txt"
        fout=open(OUTPUT,"w")        
        for tagid in self.tag_id:
            fout.write(str(self.tag_id.index(tagid))+" "+str(tagid)+"\n")
        fout.close()
        
        
        
        OUTPUT=self.path+self.data_name+outputdirectory+"/CQG.txt"
        fout=open(OUTPUT,"w")
        
        edgenum=0
        for nodeq in self.Q_asker: #write q-asker links 
            fout.write(str(nodeq)+" "+str(numq+self.Q_asker[nodeq])+" 1 \n")
            edgenum+=1
            
        for nodeq in self.Q_answerer: #write q-answerer links 
            nodeansweres=self.Q_answerer[nodeq]
            for nodea in nodeansweres:
                fout.write(str(nodeq)+" "+str(numq+nodea)+" 1 \n")
                edgenum+=1
                
        for nodeq in self.Q_tags: #write q-tag links 
            nodetags=self.Q_tags[nodeq]
            for nodet in nodetags:
                fout.write(str(nodeq)+" "+str(numq+numuser+nodet)+" 1 \n")
                edgenum+=1
        
        fout.close()
        
        #write graph into file
        N=numq+numuser+numtags
        OUTPUT=self.path+self.data_name+outputdirectory+"/CQG_proporties.txt"
        fout=open(OUTPUT,"w")
        fout.write("Num nodes= "+str(N)+" Num edges= "+str(edgenum)+"\nNum Q= "+str(numq)+" indexes=0.."+str(numq-1)
                   #+"\nNum asker= "+str(numasker)+" indexes="+str(numq)+".."+str(numq+numasker-1)
                   +"\nNum users= "+str(numuser)+" indexes="+str(numq)+".."+str(numq+numuser-1)
                   +"\nNum tags= "+str(numtags)+" indexes="+str(numq+numuser)+".."+str(numq+numuser+numtags-1))
        fout.close()
        
            
    def cleanData(self,path):
        """
        clean data: remove stop words, extract questions, answers, build train and test data
        """
        print("start")
        self.parsed=path
        outdir=self.path+self.data_name+self.parsed
        
        if not os.path.exists(outdir):
            print("{} data dir not found.\n"
              " Creating a folder for that."
              .format(outdir))
            os.mkdir(outdir)
        
        #step 1:
        self.load_data(4)# filter answers with voting scores less than 4
        
        #step 2:
        self.process_QA(min_a_size=2) # minimum number of answers for each question 2
        self.question_stats()
        self.extract_question_content()
        self.extract_answer_content()
        self.extract_answer_score()
        self.build_test_set(threshold=2, test_sample_size=5,test_proportion=0.1)    
        
        #step 4:
        #print("\tExtracting Q, R, A relations ...")
        self.extract_question_user()
        self.extract_question_answer_user()
        self.write_part_users()

        #step 6:        
        self.extract_question_best_answerer()        

        print("Cleaning data done!")  
                
    def run(self):
        #step 1:
        self.cleanData("/parsed")
        #step 2:
        self.ExtractText(20,100,0.4)      
        #step 3:
        self.extract_q_tags()        
        #step 4:
        self.extract_user_tags_score()        
        
        #step 5
        self.createGraph()        
         
        print("preprocessing data done!")
        
    def createdata(path,data):
        ob=datapreprocessing(data,path)   
        ob.run()
        
        
        
#data=["android"]  
#ob=data_preprocessing(data[0],'../data/')   
#ob.run()
        