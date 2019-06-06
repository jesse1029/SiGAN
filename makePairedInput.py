
import os
import cv2
import numpy as np
import random as rn
from Queue import Queue
import threading
import time
import signal
import sys
import pdb


fn='/home/jess/srez/CASIA/CASIA-WebFace/'
fn2 = fn+'train.txt'

with open(fn2) as fx:
	content = fx.readlines()

glen1 = len(content)
seq1 = range(0,glen1)
rn.shuffle(seq1)

targetLen = glen1*100

cnt = 0
seq2 = seq1
rn.shuffle(seq2)
balanceFactor = 0.5
posLen = targetLen*balanceFactor
pcnt = 0

fstr = ["" for x in range(glen1*100)]
tflag = 0
for i in range(glen1):
	for j in range(i+1, glen1):
		idx1 = seq1[i]
		idx2 = seq2[j]
		fn1 = content[idx1].replace('\n','')
		fn2 = fn1.split(' ')
		fnx1 = fn2[0]
		fnx1 = fnx1.replace(" ","") #Filename of full path

		fn1 = content[idx2].replace('\n','')
		fn3 = fn1.split(' ')
		fnx2 = fn3[0]
		fnx2 = fnx2.replace(" ","")
		y = 1 if int(fn2[1])==int(fn3[1]) else 0
		
		if (y==1):
			pcnt = pcnt+1
			cnt = cnt+1
			fstr[cnt] = fnx1 + " " + fnx2 + " " + str(y)
			tflag=0
		if (cnt>500) & ((cnt%5000)==0) & (tflag==0):
			tflag=1
			print("The progress is " + str(cnt  * 100.0 / (glen1*glen1))  + "% with collected " +str(cnt) + "positive samples" )
		if cnt>(targetLen/2.01):
			break
	if cnt>(targetLen/2.01):
		break
			
# Make data balance!
targetLen = pcnt*2
K = rn.randint(0, targetLen)
K2 = rn.randint(0, targetLen)

while cnt < targetLen:
	tid = K % glen1
	tid2 =K2%glen1
	
	if (tid % targetLen)==0:
		K = rn.randint(0, targetLen)
		K2 = rn.randint(0, targetLen)
	
	idx1 = seq1[tid]
	idx2 = seq2[tid2]
	fn1 = content[idx1].replace('\n','')
	fn2 = fn1.split(' ')
	fnx1 = fn2[0]
	fnx1 = fnx1.replace(" ","") #Filename of full path

	fn1 = content[idx2].replace('\n','')
	fn3 = fn1.split(' ')
	fnx2 = fn3[0]
	fnx2 = fnx2.replace(" ","")

	y = 1 if int(fn2[1])==int(fn3[1]) else 0
	if (y==0):
		fstr[cnt] = fnx1 + " " + fnx2 + " " + str(y)
		cnt = cnt+1
	
	K=K+1
	K2 = K2+1
	if (cnt>50000) & ((cnt%222222)==0):
		#pdb.set_trace()
		print("The progress is " + str(cnt  * 100.0 / targetLen)  + "%")

text_file = open("CASIA/pairwise.txt", "w")
for item in fstr:
	if item == "":
		break
	text_file.write("%s\n" % item)

text_file.close()
