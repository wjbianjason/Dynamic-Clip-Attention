#-*-coding:utf-8 -*-
import sys
import numpy as np
import pickle
from qa.data_process import Vocab
np.random.seed(1337)

class Util(object):	
	@staticmethod
	def generate_vocab(file_list,output_file,task):
		vf = open(output_file,'w')
		vocab = {}
		filenames = file_list
		for filename in filenames:
			for line in file(filename):
				sents = line.strip().split("\t")
				for i in range(2):
					words = sents[i].lower().split(" ")
					for word in words:
						if not vocab.has_key(word):
							vocab[word] = len(vocab) + 1
		if task == "wikiqa":
			for word,index in sorted(vocab.items(),key=lambda x:x[1]):
				vf.write(str(index)+" "+word+"\n")
		elif task == "trecqa":
			for word,index in vocab.items():
				vf.write(str(index)+" "+word+"\n")

	@staticmethod
	def generate_embed(vocab_file,glovec_file,output_file):
		vo = Vocab(vocab_file,800000)
		embeding_list = [[] for i in range(len(vo._word_to_id))]
		padding = np.random.randn(300) * 0.2 
		embeding_list[0] = padding
		count = 0
		for line in file(glovec_file):
			units = line.strip().split(" ")
			word = units[0].lower()
			if vo._word_to_id.has_key(word):
				vector = map(float,units[1:])
				index = vo.WordToId(word)
				if len(embeding_list[index]) == 0:
					embeding_list[index] = vector
				else:
					continue
				count += 1
		# print count
		for i in range(vo.NumIds()):
			if len(embeding_list[i]) == 0:
				temp_vec = (np.random.randn(300) * 0.2).tolist()
				embeding_list[i] = temp_vec
		embedding_vec = np.array(embeding_list)
		print embedding_vec.shape
		# print count
		embedding_vec.dump(output_file)
	
	@staticmethod
	def generate_data(input_file,vocab_file,output_file):
		vo = Vocab(vocab_file,800000)
		ff = open(output_file,"wb")
		data = []
		count = set()
		for line in file(input_file):
			units = line.lower().strip().split("\t")
			count.add(units[0])
			question = map(int,vo.Encode(units[0].split(" ")))
			answer = map(int,vo.Encode(units[1].split(" ")))
			label = int(units[2])
			data.append((question,answer,label))
		print len(count)
		pickle.dump(data,ff)

	@staticmethod
	def generate_answer(file_list,output_file):
		answer_dic = {}
		index = 0
		for filename in file_list:
			data = pickle.load(open(filename,'rb'))
			for item in data:
				if item[2] == 0:
					answer_dic.setdefault(index,item[1])
					index += 1
		pickle.dump(answer_dic,open(output_file,'wb'))


def preprocess_trecqa(input_dir,output_dir):
	def generate_combine_data(sub_dir):
	        b_file = open(input_dir+sub_dir+"/b.toks",'r')
	        s_file = open(input_dir+sub_dir+"/sim.txt",'r')
	        new_file = open(output_dir+sub_dir+".txt",'w')
	        for a_line in file(input_dir+sub_dir+"/a.toks"):
	                a_line = a_line.strip()
	                b_line = b_file.readline().strip()
	                s_line = s_file.readline().strip()
	                new_file.write(a_line+"\t"+b_line+"\t"+s_line+"\n")
	generate_combine_data("train-all")
	generate_combine_data("clean-dev")
	generate_combine_data("clean-test")


if __name__ == "__main__":
	task = sys.argv[1]
	if task == "wikiqa":
		print "generate vocab"
		Util.generate_vocab(file_list=["./data/raw_data/WikiQA/WikiQACorpus/WikiQA-train.txt","./data/raw_data/WikiQA/WikiQACorpus/WikiQA-dev.txt","./data/raw_data/WikiQA/WikiQACorpus/WikiQA-test.txt"],output_file="./data/wikiqa/vocab_wiki.txt","wikiqa")
		print "generate emb"
		Util.generate_embed(vocab_file="./data/wikiqa/vocab_wiki.txt",glovec_file="./data/glove/glove.840B.300d.txt",output_file="./data/wikiqa/wikiqa_glovec.txt")
		print "generate data pkl"		
		Util.generate_data("./data/raw_data/WikiQA/WikiQACorpus/WikiQA-train.txt","./data/wikiqa/vocab_wiki.txt","./data/wikiqa/wiki_train.pkl")
		Util.generate_data("./data/raw_data/WikiQA/WikiQACorpus/WikiQA-dev.txt","./data/wikiqa/vocab_wiki.txt","./data/wikiqa/wiki_dev.pkl")
		Util.generate_data("./data/raw_data/WikiQA/WikiQACorpus/WikiQA-test.txt","./data/wikiqa/vocab_wiki.txt","./data/wikiqa/wiki_test.pkl")
		Util.generate_answer(["./data/wikiqa/wiki_train.pkl"],"./data/wikiqa/wiki_answer_train.pkl")  # random answer from train data for batch training
	elif task == "trecqa":
		preprocess_trecqa(input_dir="./data/raw_data/TrecQA/",output_dir="./data/trecqa/")
		print "generate vocab"
		Util.generate_vocab(file_list=["./data/trecqa/train-all.txt","./data/trecqa/clean-dev.txt","./data/trecqa/clean-test.txt"],output_file="./data/trecqa/vocab_trec.txt","trecqa")
		print "generate emb"
		Util.generate_embed(vocab_file="./data/trecqa/vocab_trec.txt",glovec_file="./data/glove/glove.840B.300d.txt",output_file="./data/trecqa/trecqa_glovec.txt")
		print "generate data pkl"
		Util.generate_data("./data/trecqa/train-all.txt","./data/trecqa/vocab_trec.txt","./data/trecqa/trec_train.pkl")
		Util.generate_data("./data/trecqa/clean-dev.txt","./data/trecqa/vocab_trec.txt","./data/trecqa/trec_dev.pkl")
		Util.generate_data("./data/trecqa/clean-test.txt","./data/trecqa/vocab_trec.txt","./data/trecqa/trec_test.pkl")
		Util.generate_answer(["./data/trecqa/trec_train.pkl"],"./data/trecqa/trec_answer_train.pkl")  # random answer from train data for batch training
	else:
		sys.stderr.write("illegal param")
