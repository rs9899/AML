import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from numpy import linalg as LA
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# drive 2 walk stand walk

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

obj = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]
predicate_orig = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
predicate_zero = ["run" , "give" , "leave" , "enter" , "exit" , "draw" ,"see" , "swim" , "help" , "start" , "bring" , "write" , "meet" , "read" ,"open" , "die" , "kill" , "stop" , "teach" ,"think"]

class LossHistory(cb.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		batch_loss = logs.get('loss')
		self.losses.append(batch_loss)


def init_model():
	start_time = time.time()
	print('Compiling FNN ... ')
	model = Sequential()
	model.add(Dense(1000, input_dim=500))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(300))
	###
	rms = RMSprop()
	model.summary()
	model.compile(loss='mean_squared_error', optimizer=rms, metrics=['accuracy'])
	print('Model compiled in {0} seconds'.format(time.time() - start_time))
	return model

def run_network(data=None, model=None, epochs=20, batch=256):
	try:
		start_time = time.time()
		if data is None:
			X_train, X_test, y_train, y_test = load_data()
		else:
			X_train, X_test, y_train, y_test = data

		if model is None:
			model = init_model()

		history = LossHistory()

		print('Training model...')
		model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
				  callbacks=[history],
				  validation_data=(X_test, y_test), verbose=2)

		print("Training duration : {0}".format(time.time() - start_time))
		score = model.evaluate(X_test, y_test, batch_size=16)

		print("Network's test score [loss, accuracy]: {0}".format(score))
		return model, history.losses
	except KeyboardInterrupt:
		print(' KeyboardInterrupt')
		return model, history.losses

def predict(input,FCmodel,model):
    global predicate_orig
    global predicate_zero
    output = FCmodel.predict(input)
    predicate_out = []
    for embedding in output:
        mini =  LA.norm(embedding-model["eat"])
        predicate_min = "eat"
        for predicate in predicate_orig:
            predicate=predicate.split()[0]
            x = LA.norm(embedding-model[predicate])
            if x < mini:
                predicate_min = predicate
                mini = x
        for predicate in predicate_zero:
            predicate=predicate.split()[0]
            x = LA.norm(embedding-model[predicate])
            if x < mini:
                predicate_min = predicate
                mini = x
        predicate_out.append(predicate_min)
    return predicate_out

# def main():
def loadWordModel(filename):
	# filename = './GoogleNews-vectors-negative300.bin'
	wordModel = KeyedVectors.load_word2vec_format(filename, binary=False)
	print("--- Word-vectors loaded ----")
	return wordModel

def train(path, wordModel, model=None, epochs=2, batch=256):
	roidb_file = np.load(path,encoding='bytes')
	roidb_temp = roidb_file['roidb']
	roidb = roidb_temp[()]
	print("Data step 1")
	num_img = len(roidb[b'subVec'])
	# sub = []
	# obj = []
	# diff = []
	pred = []
	word = []
	subObj = []
	for k in range(num_img):
		# if k%10 == 0:
		# 	print(k)
		num_pred = len(roidb[b'pred_roidb'][k][b'original_pred'])
		for l in range(num_pred):
			sVec = roidb[b'subVec'][k][l]
			oVec = roidb[b'objVec'][k][l]
			predPos = int(roidb[b'pred_roidb'][k][b'original_pred'][l])
			pWord = predicate_orig[int(predPos)].split()[0]
			pVec = wordModel[pWord]
			kV = np.zeros(1000)
			kV[:500] = sVec
			kV[500:] = oVec
			# sub.append(sVec)
			# obj.append(oVec)
			subObj.append(sVec - oVec)
			# subObj.append(kV)
			pred.append(pVec)
			word.append(pWord)
	# myDict = {}
	# myDict['sub'] = sub
	# myDict['obj'] = obj
	# myDict['pred'] = pred
	inp = np.asarray(subObj)
	print(inp.shape)
	out = np.asarray(pred)
	print(out.shape)
	######
	data_length = inp.shape[0]
	split = 0.9
	part = int(split * data_length)
	nnModel , _ = run_network(data=[inp[:part,:],  inp[part:,:], out[:part,:] ,out[part:,:]], model=None, epochs=epochs, batch=batch)
	return nnModel


wordModel = loadWordModel('./glove.word2vec')
trainedModel = train('./vrd_pred_train_roidb.npz',wordModel, model=None, epochs=50, batch=32)

path='./vrd_pred_test_roidb.npz'
roidb_file = np.load(path,encoding='bytes')
roidb_temp = roidb_file['roidb']
roidb = roidb_temp[()]
print("Data step 1")
num_img = len(roidb[b'subVec'])
# sub = []
# obj = []
# diff = []
pred = []
word = []
subObj = []
for k in range(num_img):
	# if k%10 == 0:
	# 	print(k)
	p=[]
	w=[]
	s=[]
	num_pred = len(roidb[b'pred_roidb'][k][b'original_pred'])
	for l in range(num_pred):
		sVec = roidb[b'subVec'][k][l]
		oVec = roidb[b'objVec'][k][l]
		predPos = int(roidb[b'pred_roidb'][k][b'original_pred'][l])
		pWord = predicate_orig[int(predPos)].split()[0]
		pVec = wordModel[pWord]
		kV = np.zeros(1000)
		kV[:500] = sVec
		kV[500:] = oVec
		# sub.append(sVec)
		# obj.append(oVec)
		s.append(sVec - oVec)
		# s.append(kV)
		p.append(pVec)
		w.append(pWord)
	pred.append(p)
	subObj.append(s)
	word.append(w)

def tester(i,subObj,trainedModel,wordModel,word):
	inp = np.asarray(subObj[i])
	print(inp.shape)
	out = predict(inp,trainedModel,wordModel)
	for k in range(len(out)):
		print( out[k], word[i][k] )
	print()

def predict2(i,input,FCmodel,model):
	p = FCmodel.predict(input)
	for j in range(p.shape[0]):
		print(word[i][j])
		print(model.similar_by_vector(p[j]))
		print("-------")

def accuracy1(subObj,trainedModel,wordModel,word):
	correct = 0
	total = 0
	for i in range(len(subObj)):
		inp = np.asarray(subObj[i])
		out = predict(inp,trainedModel,wordModel)
		for k in range(len(out)):
			total = total + 1
			if( out[k].lower() ==  word[i][k].lower() ):
				correct = correct+1
	print(correct/total)

def accuracy2(subObj,trainedModel,wordModel,word , topn):
	correct = 0
	total = 0
	for i in range(len(subObj)):
		inp = np.asarray(subObj[i])
		out = trainedModel.predict(inp)
		for k in range(out.shape[0]):
			total = total + 1
			pred = wordModel.similar_by_vector(out[k],topn=topn)
			for j in pred:
				if( j[0].lower() ==  word[i][k].lower() ):
					correct = correct+1
					break
	print(correct/total)



# 1 is an image number in testdata RANGE 1-954
tester(1,subObj,trainedModel,wordModel,word)
predict2(1,np.asarray(subObj[1]),trainedModel,wordModel)
accuracy1(subObj,trainedModel,wordModel,word)


"""
	model.predict(FOR SOME INPUT) and then Use the nearest neighbour case ---
	>>> testInp = inp[333:334]
	>>> testWord = word[333]
	>>> testWord
	'above'
	>>> p = nnModel.predict(testInp)
	>>> wordModel.similar_by_vector(p[0])
	[('above', 0.833683967590332), ('below', 0.7182117700576782), ('on', 0.6775867342948914), ('beneath', 0.5636531710624695), ('under', 0.5418061017990112), ('beyond', 0.5360198020935059), ('in', 0.5325754284858704), ('onthe', 0.5152839422225952), ('the', 0.5117425918579102), ('behind', 0.4980994164943695)]
	>>> 


"""
# np.savez('./data.npz' , myDict = myDict )
from keras.models import model_from_json
def saver(path,model):
	model_json = model.to_json()
	with open(path + "model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(path + "model.h5")
	print("Saved model to disk")
def loader(path):
	json_file = open(path + 'model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(path + "model.h5")
	print("Loaded model from disk")
	return loaded_model

print("DONE")
print()


# main()
