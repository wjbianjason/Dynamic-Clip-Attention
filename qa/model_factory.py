#-*-coding:utf-8-*-
from keras.layers import LSTM, Dense, Activation, Dropout, Input, merge, RepeatVector, Merge, Lambda ,Flatten, BatchNormalization,Permute
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from keras.optimizers import RMSprop,Adamax,SGD,Adam
from keras.layers import Embedding

from keras import backend as K
from keras.layers.wrappers import TimeDistributed
import numpy as np
np.random.seed(1337)

class ModelFactory(object):
    @staticmethod
    def get_listwise_model(model_param, embedding_file, vocab_size):
        def get_core_model(model_param, embedding_file, vocab_size):
            class _Attention(object):
                def __init__(self, ques_length, answer_length, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
                    self.ques_length = ques_length
                    self.answer_length = answer_length
                def __call__(self, sent1, sent2, reverse = False):
                    def _outer(AB):
                        att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
                        return K.permute_dimensions(att_ji,(0, 2, 1))
                    if reverse:
                        return merge(
                            [sent2, sent1],
                            mode=_outer,
                            output_shape=(self.answer_length, self.ques_length))
                    else:
                        return merge(
                            [sent1, sent2],
                            mode=_outer,
                            output_shape=(self.ques_length, self.answer_length))
            class _SoftAlignment(object):
                def __init__(self, nr_hidden):
                    # self.max_length = max_length
                    self.nr_hidden = nr_hidden

                def __call__(self, sentence, attention, ques_len, max_length,  transpose=False):
                    def _normalize_attention(attmat):
                        att = attmat[0]
                        mat = attmat[1]
                        ques_len = attmat[2]
                        if transpose:
                            att = K.permute_dimensions(att,(0, 2, 1))
                        # 3d softmax
                        e = K.exp(att - K.max(att, axis=-1, keepdims=True))
                        g = e * ques_len
                        s = K.sum(g, axis=-1, keepdims=True)
                        sm_att = g / s
                        return K.batch_dot(sm_att, mat)
                    return merge([attention, sentence, ques_len], mode=_normalize_attention,
                                  output_shape=(max_length, self.nr_hidden)) # Shape: (i, n)


            hidden_dim = model_param.hidden_dim
            question = Input(
                shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

            question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')
            answer_len = Input(shape=(model_param.dec_timesteps,), dtype='float32', name='answer_len')

            answer = Input(
                shape=(model_param.dec_timesteps,), dtype='float32', name='answer_good_base')
            weights = np.load(embedding_file)
            weights[0] = np.zeros((weights.shape[1]))   

            QaEmbedding = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[1],
                                    weights=[weights],
                                    # dropout=0.2,
                                    trainable=False)
            question_emb = QaEmbedding(question)
            answer_emb = QaEmbedding(answer)


            ques_filter_repeat_len = RepeatVector(model_param.dec_timesteps)(question_len)
            ans_filter_repeat_len = RepeatVector(model_param.enc_timesteps)(answer_len)
            
            ans_repeat_len = RepeatVector(model_param.hidden_dim)(answer_len)
            ans_repear_vec = Permute((2,1))(ans_repeat_len)

            ques_repeat_len = RepeatVector(model_param.hidden_dim)(question_len)
            ques_repear_vec = Permute((2,1))(ques_repeat_len)

            SigmoidDense = Dense(hidden_dim,activation="sigmoid")
            TanhDense = Dense(hidden_dim,activation="tanh")

            QueTimeSigmoidDense = TimeDistributed(SigmoidDense,name="que_time_s")
            QueTimeTanhDense = TimeDistributed(TanhDense,name="que_time_t")

            AnsTimeSigmoidDense = TimeDistributed(SigmoidDense,name="ans_time_s")
            AnsTimeTanhDense = TimeDistributed(TanhDense,name="ans_time_t")


            question_sig = QueTimeSigmoidDense(question_emb)
            question_tanh = QueTimeTanhDense(question_emb)
            question_proj = merge([question_sig,question_tanh],mode="mul")


            answer_sig = AnsTimeSigmoidDense(answer_emb)
            answer_tanh = AnsTimeTanhDense(answer_emb)
            answer_proj = merge([answer_sig,answer_tanh],mode="mul")
            # question_proj = question_emb
            # answer_proj = answer_emb

            Attend = _Attention(model_param.enc_timesteps, model_param.dec_timesteps , hidden_dim, dropout=0.2)
            Align = _SoftAlignment( hidden_dim)

            ques_atten_metrics = Attend(question_proj,answer_proj)
            ans_atten_metrics = Attend(question_proj,answer_proj,reverse = True)
            print ques_atten_metrics._keras_shape



            # atten_metrics = merge([atten_metrics,ques_repeat_len],mode="mul")

            answer_align = Align(question_proj,ques_atten_metrics,ques_filter_repeat_len,model_param.dec_timesteps, transpose=True)
            question_align = Align(answer_proj,ans_atten_metrics,ans_filter_repeat_len,model_param.enc_timesteps,transpose=True)
            print answer_align._keras_shape


            #if Flag = "NN":
            # sim_input = merge([answer_proj,answer_align],mode="concat")
            # NNDense = TimeDistributed(Dense(hidden_dim,activation="relu"))
            # temp_sim_output = NNDense(sim_input)
            # print temp_sim_output._keras_shape
            ans_temp_sim_output = merge([answer_proj,answer_align],mode="mul")
            ques_temp_sim_output = merge([question_proj,question_align],mode="mul")

            ans_sim_output = merge([ans_temp_sim_output,ans_repear_vec],mode="mul")
            ques_sim_output = merge([ques_temp_sim_output,ques_repear_vec],mode="mul")

            cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=hidden_dim,
                              activation='relu',
                              border_mode='same') for filter_length in [1,2,3,4,5]]

            cnn_feature = merge([cnn(ans_sim_output) for cnn in cnns], mode='concat')
            maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
            cnn_pool = maxpool(cnn_feature)
            print cnn_pool._keras_shape

            OutputDense = Dense(hidden_dim,activation="tanh")
            feature = OutputDense(cnn_pool)

            cnns1 = [Convolution1D(filter_length=filter_length,
                              nb_filter=hidden_dim,
                              activation='relu',
                              border_mode='same') for filter_length in [1,2,3,4,5]]

            cnn1_feature = merge([cnn(ques_sim_output) for cnn in cnns1], mode='concat')
            cnn1_pool = maxpool(cnn1_feature)
            print cnn1_pool._keras_shape

            OutputDense1 = Dense(hidden_dim,activation="tanh")

            feature1 = OutputDense1(cnn1_pool)

            feature_total = merge([feature,feature1],mode='concat')

            FinalDense = Dense(hidden_dim,activation="tanh")
            feature_all = FinalDense(feature_total)

            ScoreDense = Dense(1)
            score = ScoreDense(feature_all)

            model = Model(input=[question,answer,question_len,answer_len],output=[score])
            return model
        #############################################
        #############################################
        #############################################
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

        question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')

        good_answer = Input(
            shape=(model_param.dec_timesteps,), dtype='float32', name='answer_base')
        answers = Input(
            shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answer_bad_base')

        answers_length = Input(shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answers_length')
        good_answer_length = Input(shape=(model_param.dec_timesteps,),dtype='float32', name='good_answer_len')
        
        basic_model = get_core_model(model_param,embedding_file,vocab_size)

        good_similarity = basic_model([question, good_answer, question_len,good_answer_length])
        sim_list = []
        for i in range(model_param.random_size):
            convert_layer = Lambda(lambda x:x[:,i],output_shape=(model_param.dec_timesteps,))
            temp_tensor = convert_layer(answers)
            temp_length = convert_layer(answers_length)
            temp_sim = basic_model([question,temp_tensor,question_len,temp_length])
            sim_list.append(temp_sim)
        total_sim = merge(sim_list,mode="concat")
        total_prob = Lambda(lambda x: K.log(K.softmax(x)), output_shape = (model_param.random_size, ))(total_sim)
        

        prediction_model = Model(
            input=[question, good_answer,question_len,good_answer_length], output=good_similarity, name='prediction_model')
        prediction_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        training_model = Model(
            input=[question, answers,question_len,answers_length], output=total_prob, name='training_model')
        training_model.compile(
            loss=lambda y_true,y_pred: K.mean(y_true*(K.log(K.clip(y_true,0.00001,1)) - y_pred )) , optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        return training_model, prediction_model


    @staticmethod
    def get_k_threshold_model(model_param, embedding_file, vocab_size):
        def get_core_model(model_param, embedding_file, vocab_size):
            class _Attention(object):
                def __init__(self, ques_length, answer_length, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
                    self.ques_length = ques_length
                    self.answer_length = answer_length
                def __call__(self, sent1, sent2, reverse = False):
                    def _outer(AB):
                        att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
                        return K.permute_dimensions(att_ji,(0, 2, 1))
                    if reverse:
                        return merge(
                            [sent2, sent1],
                            mode=_outer,
                            output_shape=(self.answer_length, self.ques_length))
                    else:
                        return merge(
                            [sent1, sent2],
                            mode=_outer,
                            output_shape=(self.ques_length, self.answer_length))
            class _SoftAlignment(object):
                def __init__(self, nr_hidden):
                    # self.max_length = max_length
                    self.nr_hidden = nr_hidden

                def __call__(self, sentence, attention, ques_len, max_length,  transpose=False ,Flag=False):
                    def _normalize_attention(attmat):
                        att = attmat[0]
                        mat = attmat[1]
                        ques_len = attmat[2]
                        if transpose:
                            att = K.permute_dimensions(att,(0, 2, 1))
                        # 3d softmax
                        e = K.exp(att - K.max(att, axis=-1, keepdims=True))
                        g = e * ques_len
        
                        s = K.sum(g, axis=-1, keepdims=True)
                        sm_att = g / s

                        if Flag:
                            if model_param.k_value_ans == -1:
                                threshold = 1.0/K.sum(ques_len,axis=-1,keepdims=True)
                            else:
                                threshold = model_param.k_value_ans
                        else:
                            if model_param.k_value_ques == -1:
                                threshold = 1.0/K.sum(ques_len,axis=-1,keepdims=True)
                            else:
                                threshold = model_param.k_value_ques

                        k_threshold_e = K.switch(K.lesser_equal(sm_att,threshold), 0.0, sm_att)
                        new_s = K.clip(K.sum(k_threshold_e,axis=-1,keepdims=True),0.00001,1024)
                        new_sm_att = k_threshold_e / new_s

                        return K.batch_dot(new_sm_att, mat)

                    return merge([attention, sentence, ques_len], mode=_normalize_attention,
                                  output_shape=(max_length, self.nr_hidden)) 

            hidden_dim = model_param.hidden_dim
            question = Input(
                shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

            question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')
            answer_len = Input(shape=(model_param.dec_timesteps,), dtype='float32', name='answer_len')

            answer = Input(
                shape=(model_param.dec_timesteps,), dtype='float32', name='answer_good_base')
            weights = np.load(embedding_file)
            weights[0] = np.zeros((weights.shape[1]))   

            QaEmbedding = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[1],
                                    weights=[weights],
                                    # dropout=0.2,
                                    trainable=False)
            question_emb = QaEmbedding(question)
            answer_emb = QaEmbedding(answer)


            ques_filter_repeat_len = RepeatVector(model_param.dec_timesteps)(question_len)
            ans_filter_repeat_len = RepeatVector(model_param.enc_timesteps)(answer_len)
            
            ans_repeat_len = RepeatVector(model_param.hidden_dim)(answer_len)
            ans_repear_vec = Permute((2,1))(ans_repeat_len)

            ques_repeat_len = RepeatVector(model_param.hidden_dim)(question_len)
            ques_repear_vec = Permute((2,1))(ques_repeat_len)

            SigmoidDense = Dense(hidden_dim,activation="sigmoid")
            TanhDense = Dense(hidden_dim,activation="tanh")

            QueTimeSigmoidDense = TimeDistributed(SigmoidDense,name="que_time_s")
            QueTimeTanhDense = TimeDistributed(TanhDense,name="que_time_t")

            AnsTimeSigmoidDense = TimeDistributed(SigmoidDense,name="ans_time_s")
            AnsTimeTanhDense = TimeDistributed(TanhDense,name="ans_time_t")


            question_sig = QueTimeSigmoidDense(question_emb)
            question_tanh = QueTimeTanhDense(question_emb)
            question_proj = merge([question_sig,question_tanh],mode="mul")


            answer_sig = AnsTimeSigmoidDense(answer_emb)
            answer_tanh = AnsTimeTanhDense(answer_emb)
            answer_proj = merge([answer_sig,answer_tanh],mode="mul")
            # question_proj = question_emb
            # answer_proj = answer_emb

            Attend = _Attention(model_param.enc_timesteps, model_param.dec_timesteps , hidden_dim, dropout=0.2)
            Align = _SoftAlignment( hidden_dim)

            ques_atten_metrics = Attend(question_proj,answer_proj)
            ans_atten_metrics = Attend(question_proj,answer_proj,reverse = True)
            print ques_atten_metrics._keras_shape



            # atten_metrics = merge([atten_metrics,ques_repeat_len],mode="mul")

            answer_align = Align(question_proj,ques_atten_metrics,ques_filter_repeat_len,model_param.dec_timesteps, transpose=True)
            question_align = Align(answer_proj,ans_atten_metrics,ans_filter_repeat_len,model_param.enc_timesteps,transpose=True,Flag=True)
            print answer_align._keras_shape


            #if Flag = "NN":
            # sim_input = merge([answer_proj,answer_align],mode="concat")
            # NNDense = TimeDistributed(Dense(hidden_dim,activation="relu"))
            # temp_sim_output = NNDense(sim_input)
            # print temp_sim_output._keras_shape
            ans_temp_sim_output = merge([answer_proj,answer_align],mode="mul")
            ques_temp_sim_output = merge([question_proj,question_align],mode="mul")

            ans_sim_output = merge([ans_temp_sim_output,ans_repear_vec],mode="mul")
            ques_sim_output = merge([ques_temp_sim_output,ques_repear_vec],mode="mul")

            cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=hidden_dim,
                              activation='relu',
                              border_mode='same') for filter_length in [1,2,3,4,5]]

            cnn_feature = merge([cnn(ans_sim_output) for cnn in cnns], mode='concat')
            maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
            cnn_pool = maxpool(cnn_feature)
            print cnn_pool._keras_shape

            OutputDense = Dense(hidden_dim,activation="tanh")
            feature = OutputDense(cnn_pool)

            cnns1 = [Convolution1D(filter_length=filter_length,
                              nb_filter=hidden_dim,
                              activation='relu',
                              border_mode='same') for filter_length in [1,2,3,4,5]]

            cnn1_feature = merge([cnn(ques_sim_output) for cnn in cnns1], mode='concat')
            cnn1_pool = maxpool(cnn1_feature)
            print cnn1_pool._keras_shape

            OutputDense1 = Dense(hidden_dim,activation="tanh")

            feature1 = OutputDense1(cnn1_pool)

            feature_total = merge([feature,feature1],mode='concat')

            FinalDense = Dense(hidden_dim,activation="tanh")
            feature_all = FinalDense(feature_total)

            ScoreDense = Dense(1)
            score = ScoreDense(feature_all)

            model = Model(input=[question,answer,question_len,answer_len],output=[score])
            return model
        #############################################
        #############################################
        #############################################
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

        question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')

        good_answer = Input(
            shape=(model_param.dec_timesteps,), dtype='float32', name='answer_base')
        answers = Input(
            shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answer_bad_base')

        answers_length = Input(shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answers_length')
        good_answer_length = Input(shape=(model_param.dec_timesteps,),dtype='float32', name='good_answer_len')
        
        basic_model = get_core_model(model_param,embedding_file,vocab_size)

        good_similarity = basic_model([question, good_answer, question_len,good_answer_length])
        sim_list = []
        for i in range(model_param.random_size):
            convert_layer = Lambda(lambda x:x[:,i],output_shape=(model_param.dec_timesteps,))
            temp_tensor = convert_layer(answers)
            temp_length = convert_layer(answers_length)
            temp_sim = basic_model([question,temp_tensor,question_len,temp_length])
            sim_list.append(temp_sim)
        total_sim = merge(sim_list,mode="concat")
        total_prob = Lambda(lambda x: K.log(K.softmax(x)), output_shape = (model_param.random_size, ))(total_sim)
        

        prediction_model = Model(
            input=[question, good_answer,question_len,good_answer_length], output=good_similarity, name='prediction_model')
        prediction_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        training_model = Model(
            input=[question, answers,question_len,answers_length], output=total_prob, name='training_model')
        training_model.compile(
            loss=lambda y_true,y_pred: K.mean(y_true*(K.log(K.clip(y_true,0.00001,1)) - y_pred )) , optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        return training_model, prediction_model

    @staticmethod
    def get_k_max_model(model_param, embedding_file, vocab_size):
        def get_core_model(model_param, embedding_file, vocab_size):
            class _Attention(object):
                def __init__(self, ques_length, answer_length, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
                    self.ques_length = ques_length
                    self.answer_length = answer_length
                def __call__(self, sent1, sent2, reverse = False):
                    def _outer(AB):
                        att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
                        return K.permute_dimensions(att_ji,(0, 2, 1))
                    if reverse:
                        return merge(
                            [sent2, sent1],
                            mode=_outer,
                            output_shape=(self.answer_length, self.ques_length))
                    else:
                        return merge(
                            [sent1, sent2],
                            mode=_outer,
                            output_shape=(self.ques_length, self.answer_length))
            class _SoftAlignment(object):
                def __init__(self, nr_hidden):
                    # self.max_length = max_length
                    self.nr_hidden = nr_hidden

                def __call__(self, sentence, attention, ques_len, max_length,  transpose=False,Flag=False):
                    def _normalize_attention(attmat):
                        att = attmat[0]
                        mat = attmat[1]
                        ques_len = attmat[2]
                        if transpose:
                            att = K.permute_dimensions(att,(0, 2, 1))
                        # 3d softmax
                        e = K.exp(att - K.max(att, axis=-1, keepdims=True))
                        g = e * ques_len
                        # s = K.sum(g, axis=-1, keepdims=True)
                        if Flag:
                            bound = -int(model_param.k_value_ans)
                        else:
                            bound = -int(model_param.k_value_ques)

                        k_max_e = K.T.set_subtensor(g[K.T.arange(g.shape[0]).dimshuffle(0,'x','x'),K.T.arange(g.shape[1]).dimshuffle('x',0,'x'), K.T.argsort(g)[:,:,:bound]],0.0)
                        s = K.sum(k_max_e,axis=-1,keepdims=True)
                        sm_att = k_max_e / s

                        # sm_att = g / s
                        return K.batch_dot(sm_att, mat)
                    return merge([attention, sentence, ques_len], mode=_normalize_attention,
                                  output_shape=(max_length, self.nr_hidden)) # Shape: (i, n)

            hidden_dim = model_param.hidden_dim
            question = Input(
                shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

            question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')
            answer_len = Input(shape=(model_param.dec_timesteps,), dtype='float32', name='answer_len')

            answer = Input(
                shape=(model_param.dec_timesteps,), dtype='float32', name='answer_good_base')
            weights = np.load(embedding_file)
            weights[0] = np.zeros((weights.shape[1]))   

            QaEmbedding = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[1],
                                    weights=[weights],
                                    # dropout=0.2,
                                    trainable=False)
            question_emb = QaEmbedding(question)
            answer_emb = QaEmbedding(answer)


            ques_filter_repeat_len = RepeatVector(model_param.dec_timesteps)(question_len)
            ans_filter_repeat_len = RepeatVector(model_param.enc_timesteps)(answer_len)
            
            ans_repeat_len = RepeatVector(model_param.hidden_dim)(answer_len)
            ans_repear_vec = Permute((2,1))(ans_repeat_len)

            ques_repeat_len = RepeatVector(model_param.hidden_dim)(question_len)
            ques_repear_vec = Permute((2,1))(ques_repeat_len)

            SigmoidDense = Dense(hidden_dim,activation="sigmoid")
            TanhDense = Dense(hidden_dim,activation="tanh")

            QueTimeSigmoidDense = TimeDistributed(SigmoidDense,name="que_time_s")
            QueTimeTanhDense = TimeDistributed(TanhDense,name="que_time_t")

            AnsTimeSigmoidDense = TimeDistributed(SigmoidDense,name="ans_time_s")
            AnsTimeTanhDense = TimeDistributed(TanhDense,name="ans_time_t")


            question_sig = QueTimeSigmoidDense(question_emb)
            question_tanh = QueTimeTanhDense(question_emb)
            question_proj = merge([question_sig,question_tanh],mode="mul")


            answer_sig = AnsTimeSigmoidDense(answer_emb)
            answer_tanh = AnsTimeTanhDense(answer_emb)
            answer_proj = merge([answer_sig,answer_tanh],mode="mul")
            # question_proj = question_emb
            # answer_proj = answer_emb

            Attend = _Attention(model_param.enc_timesteps, model_param.dec_timesteps , hidden_dim, dropout=0.2)
            Align = _SoftAlignment( hidden_dim)

            ques_atten_metrics = Attend(question_proj,answer_proj)
            ans_atten_metrics = Attend(question_proj,answer_proj,reverse = True)
            print ques_atten_metrics._keras_shape



            # atten_metrics = merge([atten_metrics,ques_repeat_len],mode="mul")

            answer_align = Align(question_proj,ques_atten_metrics,ques_filter_repeat_len,model_param.dec_timesteps, transpose=True)
            question_align = Align(answer_proj,ans_atten_metrics,ans_filter_repeat_len,model_param.enc_timesteps,transpose=True,Flag=True)
            print answer_align._keras_shape


            #if Flag = "NN":
            # sim_input = merge([answer_proj,answer_align],mode="concat")
            # NNDense = TimeDistributed(Dense(hidden_dim,activation="relu"))
            # temp_sim_output = NNDense(sim_input)
            # print temp_sim_output._keras_shape
            ans_temp_sim_output = merge([answer_proj,answer_align],mode="mul")
            ques_temp_sim_output = merge([question_proj,question_align],mode="mul")

            ans_sim_output = merge([ans_temp_sim_output,ans_repear_vec],mode="mul")
            ques_sim_output = merge([ques_temp_sim_output,ques_repear_vec],mode="mul")

            cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=hidden_dim,
                              activation='relu',
                              border_mode='same') for filter_length in [1,2,3,4,5]]

            cnn_feature = merge([cnn(ans_sim_output) for cnn in cnns], mode='concat')
            maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
            cnn_pool = maxpool(cnn_feature)
            print cnn_pool._keras_shape

            OutputDense = Dense(hidden_dim,activation="tanh")
            feature = OutputDense(cnn_pool)

            cnns1 = [Convolution1D(filter_length=filter_length,
                              nb_filter=hidden_dim,
                              activation='relu',
                              border_mode='same') for filter_length in [1,2,3,4,5]]

            cnn1_feature = merge([cnn(ques_sim_output) for cnn in cnns1], mode='concat')
            cnn1_pool = maxpool(cnn1_feature)
            print cnn1_pool._keras_shape

            OutputDense1 = Dense(hidden_dim,activation="tanh")

            feature1 = OutputDense1(cnn1_pool)

            feature_total = merge([feature,feature1],mode='concat')

            FinalDense = Dense(hidden_dim,activation="tanh")
            feature_all = FinalDense(feature_total)

            ScoreDense = Dense(1)
            score = ScoreDense(feature_all)

            model = Model(input=[question,answer,question_len,answer_len],output=[score])
            return model
        #############################################
        #############################################
        #############################################
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

        question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')

        good_answer = Input(
            shape=(model_param.dec_timesteps,), dtype='float32', name='answer_base')
        answers = Input(
            shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answer_bad_base')

        answers_length = Input(shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answers_length')
        good_answer_length = Input(shape=(model_param.dec_timesteps,),dtype='float32', name='good_answer_len')
        
        basic_model = get_core_model(model_param,embedding_file,vocab_size)

        good_similarity = basic_model([question, good_answer, question_len,good_answer_length])
        sim_list = []
        for i in range(model_param.random_size):
            convert_layer = Lambda(lambda x:x[:,i],output_shape=(model_param.dec_timesteps,))
            temp_tensor = convert_layer(answers)
            temp_length = convert_layer(answers_length)
            temp_sim = basic_model([question,temp_tensor,question_len,temp_length])
            sim_list.append(temp_sim)
        total_sim = merge(sim_list,mode="concat")
        total_prob = Lambda(lambda x: K.log(K.softmax(x)), output_shape = (model_param.random_size, ))(total_sim)
        

        prediction_model = Model(
            input=[question, good_answer,question_len,good_answer_length], output=good_similarity, name='prediction_model')
        prediction_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        training_model = Model(
            input=[question, answers,question_len,answers_length], output=total_prob, name='training_model')
        training_model.compile(
            loss=lambda y_true,y_pred: K.mean(y_true*(K.log(K.clip(y_true,0.00001,1)) - y_pred )) , optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        return training_model, prediction_model


    @staticmethod
    def get_model(model_param, embedding_file, vocab_size,model_type):
        if model_type == "listwise":
            return ModelFactory.get_listwise_model(model_param, embedding_file, vocab_size)
        elif model_type == "k_max":
            return ModelFactory.get_k_max_model(model_param, embedding_file, vocab_size)
        elif model_type == "k_threshold":
            return ModelFactory.get_k_threshold_model(model_param, embedding_file, vocab_size)



        





