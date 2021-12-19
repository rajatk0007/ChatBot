"""****************************************CHATBOTBOT IMPLEMENTATION************************************************"""

"""Importing the required libraries"""
import numpy as np
import tensorflow as tf
import re
import time


"""*****************************************DATA PREPROCESSING***********************************************************"""


"""Importing & splitting the lines and conversation file by each new line"""
lines = open('movie_lines.txt',encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding = 'utf-8', errors = 'ignore').read().split('\n')


"""Creating a dictionary for lines file"""
dict1 = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        dict1[_line[0]] = _line[4] #Matching the corresponding id to line


"""Creating a list of the conversations"""        
list1 = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")#only conversation ID's
    list1.append(_conversation.split(","))
    

"""Making the question and answer list"""
question = []
answer = []
for conversation in list1:
    for i in range(len(conversation)-1):
        question.append(dict1[conversation[i]])
        answer.append(dict1[conversation[i+1]])


"""Cleaning the text"""
def clean(text):
    text = text.lower()
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"it's","it is", text)
    text = re.sub(r"\'ll"," will", text)
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"\'ve"," have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re"," are", text)
    text = re.sub(r"\'s"," is", text)
    text = re.sub(r"won't","will not", text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"\W"," ",text,flags=re.I)
    text = re.sub(r"\s+"," ", text, flags = re.I)
    return text


"""Making the cleaned questions list"""
clean_question = []
for i in question:
    clean_question.append(clean(i))
    
"""Making the cleaned answers list"""
clean_answer = [] 
for j in answer:
    clean_answer.append(clean(j))
             

"""Counting word occurences"""
word_count = {}
#FOR QUESTIONS
for question in clean_question:
    for word in question.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
#FOR ANSWERS
for answer in clean_answer:
    for word in answer.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
            

            
"""Mapping each word to a unique integer"""
min_occurence = 20
question_int = {}
answer_int = {}
num = 0
for word, count in word_count.items():
    if count >= min_occurence:
        question_int[word] = num
        answer_int[word] = num
        num += 1
        

"""Adding the tokens"""
tokens = ["<start>","<end>","<pad>","<out>"]
for token in tokens:
    question_int[token] = len(question_int)+1
for token in tokens:
    answer_int[token] = len(answer_int)+1
    
    
"""Inversing the answer_int dictionary"""
int_answer = {a : b for b,a in answer_int.items()}


"""Adding <end> token to the end of each answer"""
for i in range(len(clean_answer)):
    clean_answer[i] += '<end>'
    
    
"""Changing the texts into unique integers"""
question_list = []
for question in clean_question:
    list2 = []
    for word in question.split():
        if word not in question_int:
            list2.append(question_int['<out>'])
        else:
            list2.append(question_int[word])
    question_list.append(list2)

answer_list = []
for answer in clean_answer:
    list2 = []
    for word in answer.split():
        if word not in answer_int:
            list2.append(answer_int['<out>'])
        else:
            list2.append(answer_int[word])
    answer_list.append(list2)
    
    
"""Sorting the questions and answers as per length"""
sorted_question = []
sorted_answer = []
for length in range(1,26):#taking max length of text to be 25
    for i in enumerate(question_list):
        if(len(i[1]) == length):
            sorted_question.append(question_list[i[0]])
            sorted_answer.append(answer_list[i[0]])
            


        
    
    
"""*************************************************SEQ2SEQ*MODEL************************************************"""



"""Creating the placeholders"""
def model_input():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32)
    return inputs,targets,lr,keep_prob


"""Preprocessing the targets"""
def preprocess_target(targets, word_int, batch_size):
    left_side = tf.fill([batch_size,1],word_int['<start>'])#filling the tensor with start token
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])#removing the end column from every list
    preprocessed_target = tf.concat([left_side,right_side],1)# axis=1 for horizontal concatenation
    return preprocessed_target


"""RNN->Encoder Layer"""
def encoder_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
     lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
     lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
     encoder_cell  = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
     encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                        cell_bw=encoder_cell,
                                                        sequence_length=sequence_length,
                                                        inputs=rnn_inputs,
                                                        dtype=tf.float32)
     return encoder_state
 
    
"""RNN->Decoder_Layer_for_training_set"""
def  decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size ):
     attention_state = tf.zeros([batch_size,1,decoder_cell.output_size])
     attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option="bahdanau", num_units=decoder_cell.output_size)
     training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                               attention_keys,
                                                                               attention_values,
                                                                               attention_score_function,
                                                                               attention_construct_function,
                                                                               name = "attn_dec_train")
     decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                               training_decoder_function,
                                                                                                               decoder_embedded_input,
                                                                                                               sequence_length,
                                                                                                               scope = decoding_scope)
     decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
     return output_function(decoder_output_dropout)
 
     
"""RNN->Decoder_layer_for_validation_set_and_test_set"""
def  decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix,start_id, end_id, max_length,num_words ,decoding_scope, output_function, keep_prob, batch_size ):
     attention_state = tf.zeros([batch_size,1,decoder_cell.output_size])
     attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option="bahdanau", num_units=decoder_cell.output_size)
     test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              start_id,
                                                                              end_id,
                                                                              max_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
     test_predictions,decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                               test_decoder_function,
                                                                                                               scope = decoding_scope)
     return test_predictions
 
     
"""RNN->Decoder_layer"""
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length,rnn_size,num_layers,word_int,keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        decoder_cell  = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializer=weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,decoder_cell,decoder_embeddings_matrix,word_int['<start>'],word_int['<end>'],sequence_length-1,num_words,decoding_scope,output_function,keep_prob,batch_size)
        return training_predictions,test_predictions
    
        
                     
    
    
"""Seq2Seq_Model"""
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, question_int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,answers_num_words+1,encoder_embedding_size,initializer = tf.random_uniform_initializer(0,1),)
    encoder_state = encoder_layer(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_target(targets,question_int,batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,encoder_state,questions_num_words,sequence_length,rnn_size,num_layers,question_int,keep_prob,batch_size)
    return training_predictions, test_predictions
        


        
"""******************************TRAINING*THE*MODEL**************************"""
epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_input()
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answer_int),
                                                       len(question_int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       question_int)
 
# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<pad>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, question_int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answer_int))
        yield padded_questions_in_batch, padded_answers_in_batch
 
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_question) * 0.15)
training_questions = sorted_question[training_validation_split:]
training_answers = sorted_answer[training_validation_split:]
validation_questions = sorted_question[:training_validation_split]
validation_answers = sorted_answer[:training_validation_split]
 
# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
            
    
                                                                                                                     
                                                                                                                     
    




    
    
    

        




 



