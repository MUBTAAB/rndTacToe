# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:35:37 2017

@author: mucs_b
"""
#Import
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from ast import literal_eval
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings("ignore")

#Classes, functions
class robot():
        def __init__(self
                     ,player = 1
                     ,name = 'unnamed'
                     ,model = MLPClassifier(solver='lbfgs')
                     ,test_size = 0.33
                     ,random_chance = 0):
            
            self.player = player
            self.name = name
            self.model = model 
            self.test_size = test_size
            self.random_chance = random_chance
            
            self.positions = [(0,0),(0,1),(0,2)
                            ,(1,0),(1,1),(1,2)
                            ,(2,0),(2,1),(2,2)]      
        
        def predict(self, X):
            rnd = random.random()
            
            if rnd <= self.random_chance:
                return([str(random.choice(self.positions))])
            
            return(self.model.predict(X))
            
        def learn(self, frame, verbose = 0):
            frame = frame.loc[(frame['winner'] == self.player) & (frame['player'] == self.player)]
            X = frame.iloc[:,0:-4]
            y = frame['step']
            

            self.model.fit(X,y)
            if verbose == 1:
                ypred = self.model.predict(X)
                print(np.mean(y == ypred))

def checkwin(iPlayer,Coordinates,ipositions,frame, game = 0):
    
    cond1 = Coordinates[0][0]+Coordinates[0][1]+Coordinates[0][2] == iPlayer.player*3
    cond2 = Coordinates[1][0]+Coordinates[1][1]+Coordinates[1][2] == iPlayer.player*3
    cond3 = Coordinates[2][0]+Coordinates[2][1]+Coordinates[2][2] == iPlayer.player*3
        
    cond4 = Coordinates[0][0]+Coordinates[1][0]+Coordinates[2][0] == iPlayer.player*3
    cond5 = Coordinates[0][1]+Coordinates[1][1]+Coordinates[2][1] == iPlayer.player*3
    cond6 = Coordinates[0][2]+Coordinates[1][2]+Coordinates[2][2] == iPlayer.player*3    
        
    cond7 = Coordinates[0][0]+Coordinates[1][1]+Coordinates[2][2] == iPlayer.player*3
    cond8 = Coordinates[0][2]+Coordinates[1][1]+Coordinates[2][0] == iPlayer.player*3
    
    if any([cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8]):
        frame['winner'] = iPlayer.player
        frame['game'] = game
        return(iPlayer.player)
    

    
    if len(ipositions) == 0:
        frame['winner'] = 0
        frame['game'] = game
        return(0)
    else:
        return
    
def play(player1, player2, verbose = 0, vis = 0, game = 0, record = False):
    
    positions = [(0,0),(0,1),(0,2)
                ,(1,0),(1,1),(1,2)
                ,(2,0),(2,1),(2,2)]
        
    strpos = [str(i) for i in positions]
    
    lCord = [[0,0,0],
             [0,0,0],
             [0,0,0]]        
    
    current_player = player1
    stp = 0
    iRnd = 0
    
    dfinit = pd.DataFrame()
    steps = []
    players = []
    
    while len(positions) > 0:
        sRnd = ''
        lCord_flattened = [val for sublist in lCord for val in sublist]
        
        step = literal_eval(current_player.predict(lCord_flattened)[0])
        
        if step not in positions:
            step = random.choice(positions)
            sRnd = ' (random)'
            iRnd += 1
            
        stp += 1
        if verbose == 1:
            print(stp)
            print(current_player.name + sRnd)
            print(lCord)
            print(step)
            print('')
            
        if vis == 2:
            plt.figure(figsize = (2,2))
            img2 = plt.imshow(lCord, cmap = 'plasma')
         
        lCord_flattened = [[val] for sublist in lCord for val in sublist]
        dfinit = dfinit.append(pd.DataFrame(dict(zip(strpos,lCord_flattened))))  
        
        lCord[step[0]][step[1]] = current_player.player
        players.append(current_player.player)
        steps.append(str(step))
        
        
        if checkwin(current_player,lCord, positions, dfinit, game) != None:
            
            if vis == 1:
                plt.figure(figsize = (2,2))
                img2 = plt.imshow(lCord, cmap = 'plasma')
            if record == False:
                
                return(checkwin(current_player,lCord, positions, dfinit, game))
            
            dfinit['player'] = players
            dfinit['step'] = steps
            return(dfinit)
        
        positions.remove(step)
        
        if current_player == player1:
            current_player = player2
        else:
            current_player = player1
    
    if vis == 1:
                plt.figure(figsize = (2,2))
                img2 = plt.imshow(lCord, cmap = 'plasma') 
    
    dfinit['player'] = players
    dfinit['step'] = steps
    dfinit['winner'] = 0
    dfinit['game'] = game
    if record == False:
        return(checkwin(current_player,lCord, positions, dfinit, game))
    return(dfinit)
    

def match(player1,player2,n = 1000, record = False):
    
    if record == False:
        return([play(player1,player2) for i in range(n)])
    
    if record == True:
        dfMatch = pd.DataFrame()
        for i in range(n):
            dfMatch = dfMatch.append(play(player1,player2, record = True, game = i))
        return(dfMatch)

#Define random bots
RND1 = robot(name = 'RND1'
              ,player = 1
              ,model = MLPClassifier()
              ,random_chance = 1
              )

RND2 = robot(name = 'RND2'
              ,player = -1
              ,model = MLPClassifier()
              ,random_chance = 1
              )

#Record games of random bots to create an initial train data for your real bots 
df_start = match(RND1, RND2, n = 1000, record = True)

#Define your real bots and train them on your initial train data
Arthur = robot(name = 'Arthur'
              ,player = 1
              ,model = MLPClassifier(warm_start=1)
              ,random_chance = 0.2
              )
Arthur.learn(df_start)

King_Arthur = robot(name = 'King_Arthur'
              ,player = -1
              ,model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9,18,9), warm_start=1)
              ,random_chance = 0.1
              )
King_Arthur.learn(df_start)

#Now your bots can play
play(Arthur,King_Arthur,vis = 1)

#Make your bots play eachother, record their games and retrain them after every 500 games. 
#See their progress
score = []
for i in range(100):
    df_match = match(Arthur, King_Arthur, n = 500, record = True)
    score.append(np.mean(match(Arthur, King_Arthur,n = 100, record = False)))
    Arthur.learn(df_match, verbose = 0)
    King_Arthur.learn(df_match, verbose = 0)
plt.plot(score)
