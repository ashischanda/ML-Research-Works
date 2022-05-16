# Many to One network

# language model: https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
#                 https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
# old link:       https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/

# Input:  filename of sequence_file ("space" separated line of function_ID)
# Output: LSTM model, Heatmap of predicted function probability
# Functions: loadingFile(), model_lstm(), predict_data(), getword(), generate_seq(), predict_data_for_max



from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping

from collections import Counter
from collections import defaultdict
import operator
import seaborn as sb
import matplotlib.pyplot as plt
import pickle

# *************** PAGE/FUNCTION DICTIONARY
#id2page = pickle.load( open("/home/ashis/ASHIS/4.1_Cicero/data/pickle/id2page_new.p", "rb") )
id2page = pickle.load( open("/home/ashis/ASHIS/4.1_Cicero/data/new_data1/dictionary/full_id2page.p", "rb"))
    

#file_name = "/home/ashis/ASHIS/4.1_Cicero/code/-seq_analysis/new_claim_sequences/original_claim/cluster_5.txt" 
#file_name = "/home/ashis/ASHIS/4.1_Cicero/code/-seq_analysis/new_claim_sequences/removed_duplicate/cluster_5.txt" 
#file_name ="/home/ashis/ASHIS/4.1_Cicero/data/new_data/claim all page seq v2 RemovedDuplicate.txt"
#file_name="/home/ashis/ASHIS/4.1_Cicero/code/-seq_analysis/new_order_sequences/all.txt"
#New File
file_name ="/home/ashis/ASHIS/4.1_Cicero/data/new_data1/seq/Claim_seq_original.txt"
model_name = 'model_new_data_allClaim.h5'
file_name_w2v = "/home/ashis/ASHIS/4.1_Cicero/code/-new_page/claim_page_feature10.w2v"

top_k = 20         # setting top page number for selection
RATIO = 0.90       # train data ratio
EPOCHS = 1
END_SYMBOL = "1000"    # ADDING END SYMBOL AT THE END OF EACH LINE
SELECTED_MAX_SENTENCE_LENGTH = 80
bool_avg = False       # taking average length of input file as SELECTED_MAX_SENTENCE_LENGTH

# *****************************************************************************
# *****************************************************************************
def loadingFile(sentence, file_name):
    read = open(file_name, "r")
    st = ""
    for line in read:  # converting string into list of words      
        st += line +"\n"
        
    return st
# *****************************************************************************


sentences = []
data = loadingFile(sentences, file_name)    
# *****************************************************************************
char_counts = Counter()
for line in data.split('\n'):
    tem = line.split(" ")
    
    for t in tem:
        if len(t) > 0:
            char_counts[ t ] +=1 

# *****************************************************************************
# *********************** taking top frequent char
sorted_x1 = sorted( char_counts.items(), key=operator.itemgetter(1), reverse=True) # getting frequent starting node
sorted_x1 = sorted_x1[: top_k ]
top_char = []
for x in sorted_x1:
    top_char.append( x[0] )
    

# *****************************************************************************    
top_class_char = sorted_x1[0][0]
# *****************************************************************************


# *****************************************************************************    
# *********************** rewrite the dataset based on only top frequent char
new_data = ""
for line in data.split('\n'):
    tem = line.split(" ")
    
    if len(tem)<2:
        continue     # ignoring empty lines
        
    st = ""
    for t in tem:
        if t in top_char:
            st+= t + " "
            
    new_data += st +"\n" 

# *****************************************************************************
# *********************** remove duplicate characters
# *********************** adding end char
# *****************************************************************************
id2page[ int( END_SYMBOL )] = "END"
new_data_no_duplicate = ""
l =0
length_list = []
for line in new_data.split('\n'):
    tem = line.split(" ")
    l+=1
    prev = tem[0]
    st = prev + " "
    
    for  i in range(len( tem) ):
        if tem[i] == prev:
            continue
        else:
            
            st+= tem[i] + " "
            prev = tem[i]
            
    new_data_no_duplicate += st + END_SYMBOL +" "+ "\n" 
print('Total Sentences: %d' % l)

# *****************************************************************************
length_list = []
new_data_tem = ""
for line in new_data_no_duplicate.split('\n'):
    tt = line.split(" ")
    if len(tt)<2:
        continue
    new_data_tem += line +"\n"
    length_list.append( len(tt) )
print ("AVG "+ str( sum(length_list)/len(length_list) ))    


if bool_avg:
    avg = sum(length_list)/len(length_list)
else:    
    avg = SELECTED_MAX_SENTENCE_LENGTH
    
# *****************************************************************************
# *********************************  selecting lines that are less than avg ***
l=0
new_data_tem_tem = ""
for line in new_data_tem.split('\n'):
   
    tt = line.split(" ")
    if len(tt)>= avg:
        continue
    new_data_tem_tem += line +"\n"
    l+=1
print ("Total sequences (final) (after selecting SELECTED_MAX_SENTENCE_LENGTH): " + str(l) )


# *****************************************************************************
# **************************  shuffling data
# *****************************************************************************
new_data_list = []
for line in new_data_tem_tem.split('\n'):
    new_data_list.append( line)
    
import random
random.shuffle( new_data_list)

# *****************************************************************************
# ****************************************** spliting data into train and test
# *****************************************************************************
train_n = l * RATIO
test_n = l- train_n

train_data = ""
test_data = ""
ii =0
for line in new_data_list:    
    if ii < train_n:
        train_data+= line +"\n"
    else:
        test_data += line+"\n"
    ii+=1
# *****************************************************************************

# *****************************************************************************
# making vocabulary, dict for voca
# *****************************************************************************
tokenizer = Tokenizer()
tokenizer.fit_on_texts([ new_data_no_duplicate ])
vocab_size = len(tokenizer.word_index) +1 
print('Vocabulary Size: %d' % vocab_size)


# *********************************  Sorting page_names in order
words = [] # labels for x-axis

for word, ind in tokenizer.word_index.items():
    words.append( int( word) )  # converting into int, because we need to sort
    
w = sorted(words)
word2ind = dict()
ind2word = dict()
i =1
for word in w:
    word2ind[ word ] = i
    ind2word[ i    ] = word
    i+=1

def myTexts_to_sequences(line):
    words = line.split(" ")
    encode = []
    for w in words:
        if len(w)>0:
            word_int = int(w)
            encode.append( word2ind[ word_int ] )
        
    return encode
# *****************************************************************************
# ************************************** encoding data into token/voca index    
sequences = list()
for line in train_data.split('\n'):
    encoded = myTexts_to_sequences(line)
    for i in range(1, len(encoded) ):
        
        sequence = encoded[ :i+1]
        sequences.append(sequence)
    
print('Total Sequences: %d' % len(sequences))
# *****************************************************************************
# pad input sequences
max_length = max([len(seq) for seq in sequences])

sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)


# *****************************************************************************
train = sequences
sequences = array(train)
X, y = sequences[:,:-1],sequences[:,-1]         ## split into input and output elements
y = to_categorical(y, num_classes=vocab_size)
# **************************************************  checking top char for baseline
top_class_word_index= word2ind[ int( top_class_char) ]
# *****************************************************************************



# *****************************************************************************
# **************************  Model   *****************************************
# *****************************************************************************
def model_lstm(MODEL_INPUT):
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=MODEL_INPUT ))
        
    # ************  setting pre-trained word embedding model
    #embedding_matrix, dimension = load_embedding(file_name_w2v, tokenizer,  vocab_size)
    #model.add( Embedding(vocab_size, dimension, weights=[embedding_matrix], input_length= MODEL_INPUT , trainable=False) )
    # ************* Here, trainable=False  It means, EMBEDDING IS FIXED     
    #model.add(OneHot(input_dim=vocab_size, input_length=MODEL_INPUT))
    
    model.add( LSTM(50) )
    model.add(  Dense(vocab_size, activation='softmax') )
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # You can try other structure
    #https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
    #model.add(LSTM(50, return_sequences = True))
    #model.add(LSTM(30))
    
    return model
# *****************************************************************************

MODEL_INPUT = max_length-1   #last value is used for output, so -1
model = model_lstm(MODEL_INPUT)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
model.fit(X, y, epochs= EPOCHS, verbose=1, callbacks=[earlystop], validation_split=0.1 )
# *****************************************************************************





# *****************************************************************************
# generate a sequence from the model
def predict_data(model, tokenizer, seed_text, max_length, top_class_word_index, row_num):
    y_axis_labels = ['unk'] # labels for x-axis
    
    for word, ind in word2ind.items():
        y_axis_labels.append( id2page[ int(word) ]+" :"+ str(word) )
    
    
    acc = []
    acc_b=[]
    ii = 0
    
    
    for line in seed_text.split('\n'):
        x_axis_labels = [] # labels for xaxis
        
        
        sequences = list()
        encoded = myTexts_to_sequences(line)
        for i in range(1, len(encoded) ):
            
            sequence = encoded[ :i+1]
            sequences.append(sequence)
                
        # pad input sequences
        sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
        sequences = array(sequences)
        if len(sequences)<1:
            continue
        
        X, y = sequences[:,:-1],sequences[:,-1]         ## split into input and output elements
        yhat = model.predict_classes( X, verbose=0)
        
        
        count =0
        count_base = 0
   
        for i in range( len(y )):
            if y[i] == yhat[i]:
                count +=1
            if y[i] == top_class_word_index:
                count_base +=1
                
        
        print (ii)
        
        
        print ("Test data: (pair) " + str( len(y)) )       
        print ("Accuracy: "+ str( count /len(y)) )
        print ("Baseline: "+ str( count_base /len(y)) )
        
        acc.append( count/len(y) )
        acc_b.append( count_base/len(y) )
        
        ii+=1
        for yi in y:
            x_axis_labels.append( ind2word[ yi ] )
        if ii == row_num:
            print (yhat)
            for yyy in yhat:
                print ( ind2word[ yyy ] )
                
            yhat_prob = model.predict( X, verbose=0)
            reshape = yhat_prob.T
            heat_map = sb.heatmap( reshape, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="Blues") #cmap="cubehelix", "Blues_r" in reverse order
            #https://likegeeks.com/seaborn-heatmap-tutorial/
            
            heat_map.set(xlabel='Sample page sequence', ylabel='Page Index')
            
            plt.show()
            plt.savefig("heatmap n classes.png")
            break
        
    print ("Accuracy: "+ str( sum(acc)/len(acc) ) )
    print ("Baseline: "+ str( sum(acc_b) /len(acc_b)) )
# *****************************************************************************  

# ***************************************************************************** 
# *****************************************************************************  
# generate a sequence from the model
def getword(indeces):
    word_labels = ""
    for yi in indeces:
        word_labels+= str( ind2word[ yi ] ) +", "
    return word_labels            
# ***************************************************************************** 
def generate_seq(model, tokenizer, seed_text, given_length):
    for line in seed_text.split('\n'):
            
            
            encoded = myTexts_to_sequences(line)
            if len( encoded) < given_length+1:
                continue
            
            show_length = len(encoded)
            
            sequence = encoded[ : given_length+1]
            #generator= sequence
            
            for ii in range(given_length+1, show_length):
                sequences = list()
                sequences.append(sequence)        
                # pad input sequences
                sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
                sequences = array(sequences)
           
                
                X, y = sequences[:,:-1],sequences[:,-1]         ## split into input and output elements
                yhat = model.predict_classes( X, verbose=0)
                
                sequence.append(yhat[0])
                #generator.append(yhat)
                
            print ("-----")
            print ( getword(encoded) )
            print ( getword(sequence))
        

# ***************************************************************************** 
# ***************************************************************************** 
row_num_to_plot = 519
predict_data(model, tokenizer, test_data, max_length, top_class_word_index , row_num_to_plot)
# *****************************************************************************  


#given_length = 5
#generate_seq(model, tokenizer, test_data, given_length)

#import h5py
#from keras.models import load_model
model.save(model_name)  # pip install h5py
