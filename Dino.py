import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import io
import time
import os
import pickle
#import pickel
import gzip, pickle
import random
import base64
import json

from IPython import get_ipython
from PIL import Image
from IPython.display import clear_output
from random import randint
from io import BytesIO
from matplotlib import pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
from collections import deque

game_url = "chrome://dino"
chrome_driver_path = "../chromedriver"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"

class Game:
	def __init__(self, custom_config=True):
		chrome_options = Options()
		chrome_options.add_argument("disable-infobars")
		chrome_options.add_argument("--mute-audio")
		self._driver = webdriver.Chrome(executable_path = chrome_driver_path,chrome_options=chrome_options)
		self._driver.set_window_position(x=-10,y=0)
		self._driver.get('chrome://dino')
		self._driver.execute_script("Runner.config.ACCELERATION=0")
		self._driver.execute_script(init_script)
	def get_crashed(self):
		return self._driver.execute_script("return Runner.instance_.crashed")
	def get_playing(self):
		return self._driver.execute_script("return Runner.instance_playing")
	def restart(self):
		self._driver.execute_script("Runner.instance_.restart()")
	def press_up(self):
		self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
	def get_score(self):
		score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
		score = ''.join(score_array)
		return int(score)
	def pause(self):
		return self._driver.execute_script("return Runner.instance_.stop()")
	def resume(self):
		return self._driver.execute_script("return Runner.instance_.play()")
	def end(self):
		self._driver.close()

class DinoAgent:
    def __init__(self,game):
        self._game = game; 
        self.jump();
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def jump(self):
        self._game.press_up()
    def duck(self):
        self._game.press_down()
		
class Game_sate:
    def __init__(self,agent,game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__() 
    def get_state(self,actions):
        actions_df.loc[len(actions_df)] = actions[1]
        score = self._game.get_score() 
        reward = 0.1
        is_over = False
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver) 
        self._display.send(image)
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over
		
def save_obj(obj, name ):
    with open('objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
		
def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image	
		
def process_img(image):    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:300, :500] 
    image = cv2.resize(image, (80,80))
    return  image
		
def show_img(graphs = False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
        imS = cv2.resize(screen, (800, 400)) 
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break	



loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns =['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns = ['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns = ['actions'])
q_values_df =pd.read_csv(actions_file_path) if os.path.isfile(q_value_file_path) else pd.DataFrame(columns = ['qvalues'])
		
ACTIONS = 2
GAMMA = 0.99
OBSERVATION = 100.
EXPLORE = 100000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 16
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows , img_cols = 80,80
img_channels = 4
		
def init_cache():
    save_obj(INITIAL_EPSILON,"epsilon")
    t = 0
    save_obj(t,"time")
    D = deque()
    save_obj(D,"D")

init_cache()		
		
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8,8),padding='same',strides=(4,4),input_shape=(img_cols,img_rows,img_channels)))  #80*80*4
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    if not os.path.isfile(loss_file_path):
        model.save_weights('model.h5')
    print("We finish building the model")
    return model
		
def trainNetwork(model,game_state,observe=False):
    last_time = time.time()
    D = load_obj("D")
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] =1
    x_t, r_0, terminal = game_state.get_state(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    initial_state = s_t
    if observe :
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon") 
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
    t = load_obj("time")
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if  random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q 
                a_t[action_index] = 1 
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 
        x_t1, r_t, terminal = game_state.get_state(a_t)
        print('fps: {0}'.format(1 / (time.time()-last_time)))
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if t > OBSERVE: 
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((inputs.shape[0], ACTIONS))
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                inputs[i:i + 1] = state_t    
                targets[i] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        s_t = initial_state if terminal else s_t1
        t = t + 1
        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause() #pause game while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_obj(D,"D") #saving episodes
            save_obj(t,"time") #caching time steps
            save_obj(epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv("./objects/loss_df.csv",index=False)
            scores_df.to_csv("./objects/scores_df.csv",index=False)
            actions_df.to_csv("./objects/actions_df.csv",index=False)
            q_values_df.to_csv(q_value_file_path,index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            game_state._game.resume()
        #print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
    print("Episode finished!")
    print("************************")		
		
def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino,game)    
    model = buildmodel()
    try:
        trainNetwork(model,game_state,observe=observe)
    except StopIteration:
        game.end()

#get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (15, 9)

start = 0
interval = 10
'''
scores_df = pd.read_csv("./objects/scores_df.csv")
mean_scores = pd.DataFrame(columns =['score'])
actions_df = pd.read_csv("./objects/actions_df.csv")
max_scores = pd.DataFrame(columns =['max_score'])
q_max_scores = pd.DataFrame(columns =['q_max'])
while interval <= len(scores_df):
    mean_scores.loc[len(mean_scores)] = (scores_df.loc[start:interval].mean()['scores'])
    max_scores.loc[len(max_scores)] = (scores_df.loc[start:interval].max()['scores'])
    start = interval
    interval = interval + 10

q_max_df = pd.read_csv("./objects/q_values.csv")
'''
start = 0
interval = 1000
'''while interval <=len(q_max_df):
    q_max_scores.loc[len(q_max_scores)] = (q_max_df.loc[start:interval].mean()['actions'])
    start = interval
    interval = interval + 1000
  '''  
#mean_scores.plot()
#max_scores.plot()
#q_max_scores.plot()


		
playGame(observe=False);
