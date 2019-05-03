import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# drive 2 walk stand walk

predicate_orig = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
predicates_zero = ["run" , "give" , "leave" , "enter" , "exit" , "draw" ,"see" , "swim" , "help" , "start" , "bring" , "write" , "meet" , "read" ,"open" , "die" , "kill" , "stop" , "teach" ,"think"]

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
	model.add(Dense(200, input_dim=500))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))
	model.add(Dense(300))

	rms = RMSprop()
	model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
	print('Model compield in {0} seconds'.format(time.time() - start_time))
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


def main():
	
	filename = 'GoogleNews-vectors-negative300.bin'
	model = KeyedVectors.load_word2vec_format(filename, binary=True)
	print("--- GoogleNews-vectors-negative300 loaded ----")
	path='./vrd_pred_test_roidb.npz'
	roidb_file = np.load(path,encoding='bytes')
	roidb_temp = roidb_file['roidb']
	roidb = roidb_temp[()]
	print("Data step 1")
	num_img = len(roidb[b'subVec'])
	sub = []
	obj = []
	diff = []
	pred = []
	for k in range(num_img):
		if k%10 == 0:
			print(k)
		num_pred = len(roidb[b'pred_roidb'][k][b'original_pred'])
		for l in range(num_pred):
			sVec = roidb[b'subVec'][k][l]
			oVec = roidb[b'objVec'][k][l]
			
			predPos = int(roidb[b'pred_roidb'][k][b'original_pred'][l])
			pWord = predicate_orig[int(predPos)].split()[0]
			pVec = model[pWord]

			sub.append(sVec)
			obj.append(oVec)
			diff.append(sVec - oVec)
			pred.append(pVec)

	# myDict = {}
	# myDict['sub'] = sub
	# myDict['obj'] = obj
	# myDict['pred'] = pred

	inp = np.asarray(diff)
	print(inp.shape)
	out = np.asarray(pred)
	print(out.shape)

	data_length = inp.shape[0]
	split = 0.9
	part = int(split * data_length)
	run_network(data=[inp[:part,:].transpose() ,  inp[part:,:].transpose(), out[:part,:].transpose() ,out[part,:].transpose()], model=None, epochs=2, batch=256)

	# np.savez('./data.npz' , myDict = myDict )



	print("DONE")
	print()


main()
