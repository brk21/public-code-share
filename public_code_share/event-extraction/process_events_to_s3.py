#import webhoseio
#webhoseio.config(token="b2e1460b-80ec-49a9-8efc-3288291820c7")
import pandas as pd
import datetime
import os, sys
import time
import spacy
nlp = spacy.load('en')
import nltk.data
from nltk.tokenize import sent_tokenize
import re
import numpy as np
import itertools
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
import time
import json
import numpy as np
import boto
import boto.s3
from configparser import ConfigParser
from boto.s3.key import Key
from utils import (neighborhood_long, neighborhood_short, unix_time_millis, pretty)
from create_regexes import (dates_include_regex, dates_dont_include_regex, event_regex, info_regex)
from upload_to_s3 import (upload_to_s3)


def get_ini_vals(ini_file, section):
    config = ConfigParser()
    config.read(ini_file)
    return config[section]

aws_creds = get_ini_vals('config/config.ini', 'aws.creds')

info_records = []
event_records = []
tb = Blobber(pos_tagger=PerceptronTagger())

today = datetime.datetime.now().strftime('%Y_%m_%d%I_')

start_time = time.time()

bad_verbs = ['VBD', 'VBN']
good_verbs = ['VB', 'VBG', 'VBP', 'MD','VBZ']

LOCAL_PATH = '/home/ec2-user/tmp/'

AWS_ACCESS_KEY_ID = aws_creds['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = aws_creds['AWS_SECRET_ACCESS_KEY']
bucket_name = 'event-extraction'

files_processed = 0

# connect to the bucket
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
bucket = conn.get_bucket(bucket_name)

# # go through the list of files
bucket_list = bucket.list()
for l in bucket_list:
    keyString = str(l.key)
    d = LOCAL_PATH + keyString
    try:
        l.get_contents_to_filename(d)
    except OSError:
    # check if dir exists
        if not os.path.exists(d):
            os.makedirs(d) 
    if d.endswith('.txt'):
        file_name = os.path.abspath(d) # get full path of files
        with open(file_name, 'r') as in_file:
            print('Opened file: ',file_name)
            text = in_file.read()
            ### TOKENIZE THE TEXT
            sents = sent_tokenize(text.replace('\n','. ').replace(';','. ').replace('<br>',' ').replace(' - ','. ').replace('–',' '))
            ### LONG SENTENCES IMPLY THAT TOKENIZER MISSED SENTENCE BREAKS IN NUMBERED LISTS
            if any(len(t) > 1000 for t in sents):
                sents = sent_tokenize(text.replace('\n','. ').replace(';','. ').replace('<br>',' ').replace(' - ','. ').replace(' 1 ','. ').replace(' 2 ','. ').replace(' 3 ','. ').replace(' 4 ','. ').replace(' 5 ','. '))
            
            title = sents[0]
            
            organizations = []
            
            doc = nlp(' '.join(sents[:3]))
            
            ### EXTRACT ORGANIZATIONS
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    organizations.append(ent.text)
            ### IF NO ORGANIZATIONS, POSSIBLE THAT SPACY MISTAKES PERSON FOR ORGANIZATION
            if len(organizations) == 0:
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        organizations.append(ent.text)
            ### ITERATE THROUGH EACH SENTENCE
            ### EXTRACT RELEVANT ENTITIES FOR MECHANICAL TURKS TO VALIDATE
            for prev_sent, sentence, next_sent in neighborhood_short(sents):
                record_time = time.time()
                sent_next = str(sentence) + ' ' + str(next_sent)  
                full_context = str(prev_sent) + ' ' + str(sentence) + ' ' + str(next_sent)
                no_matches = re.findall(dates_dont_include_regex, sentence)
                info_matches = list(set(re.findall(info_regex, sentence)))
                event_matches = list(set(re.findall(event_regex, sentence)))

                if info_matches and not no_matches:
                    date_matches = list(set(re.findall(dates_include_regex, full_context)))
                    tagged = tb(sentence)
                    tag_score = 0.
                    if tagged.tags:
                        tag_score = np.mean([1. for x in tagged.tags if x[1] in good_verbs] + [0. for x in tagged.tags if x[1] in bad_verbs] + [0.])

                    new_record = {
                        'companies': '; '.join(list(set(organizations))),
                        'info_matches': info_matches,
                        'verb_tense_score': tag_score,
                        'timing_word_matches': date_matches,
                        'article_title': sents[0],
                        'article_context': full_context,
                        'article_id': d
                    }

                    info_records.append(new_record)
                    if len(info_records) % 10 == 0:
                        print('Info Match: ')
                        print('Record Time: ',(time.time() - record_time))
                        pretty(new_record)

                if event_matches and not no_matches:
                    date_matches = list(set(re.findall(dates_include_regex, full_context)))
                    if date_matches:
                        tagged = tb(full_context)
                        tag_score = 0.
                        if tagged.tags:
                            tag_score = np.mean([1. for x in tagged.tags if x[1] in good_verbs] + [0. for x in tagged.tags if x[1] in bad_verbs] + [0.])

                        new_record = {
                            'companies': '; '.join(list(set(organizations))),
                            'event_matches': event_matches,
                            'verb_tense_score': tag_score,
                            'timing_word_matches': date_matches,
                            'article_title': sents[0],
                            'article_context': full_context,
                            'article_id': d
                        }

                        event_records.append(new_record)
                        if len(event_records) % 10 == 0:
                            print('Event Match: ')
                            print('Record Time: ',(time.time() - record_time))
                            pretty(new_record)

            files_processed += 1
            if files_processed % 100 == 0:
                print("Documents Reviewed: ", files_processed)
                print("--- %s seconds for last 100 records ---" % (time.time() - start_time))
                pretty(new_record)
                start_time = time.time()
                info_df = pd.DataFrame(info_records)
                event_df = pd.DataFrame(event_records)

                info_file_name = 'info_records_' + today + str(files_processed) + '.csv'
                event_file_name = 'event_records_' + today + str(files_processed) + '.csv'

                info_df.to_csv(LOCAL_PATH + info_file_name)
                event_df.to_csv(LOCAL_PATH + event_file_name)

                try:
                    upload_to_s3(bucket_name,info_file_name,LOCAL_PATH,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
                    os.remove(LOCAL_PATH + info_file_name)
                    print("File removed: ", info_file_name)
                    upload_to_s3(bucket_name,event_file_name,LOCAL_PATH,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
                    os.remove(LOCAL_PATH + event_file_name)
                    print("File removed: ", event_file_name)
                    today = datetime.datetime.now().strftime('%Y_%m_%d%I_')
                    info_records = []
                    event_records = []
                except:
                    continue
        ### CLEAN UP FILES
        os.remove(file_name)
