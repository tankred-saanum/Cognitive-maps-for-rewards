import viz
import viztask
import vizact
import vizinfo
import vizproximity
import vizshape
import vizcam
import numpy as np
import vizinput
import os
import time
import json
import math
from ast import literal_eval

global stimDir
global test

rootDir = 'C:/Users/user/Documents/ChoiceMaps/experiment/version_scan/'

test = False

viz.fov(60)
viz.setMultiSample(4)

myScene1 = viz.addScene()
myScene2 = viz.addScene()
viz.MainView.setScene(myScene2)

if test == True:
	screensize = [1680,1050]
	viz.go(viz.FULLSCREEN)
	
#	viz.go(viz.FULLSCREEN)
#	screensize = [1680,1050]
	
#	screensize = [2560,1440]
#	viz.go(viz.FULLSCREEN)
else:
	screensize = [1680,1050]
	viz.go(viz.FULLSCREEN)

viz.mouse.setVisible(viz.OFF)
viz.antialias = 4

#######################
## PROXIMITY MANAGER ##
#######################

#Create proximity manager and set debug on. Toggle debug with d key
global manager
manager = vizproximity.Manager()
manager.setDebug(viz.OFF)
debugEventHandle = vizact.onkeydown('d',manager.setDebug,viz.TOGGLE)

#Add main viewpoint as proximity target
global target
target = vizproximity.Target(viz.MainView)
manager.addTarget(target)

###################
##OPEN DATA FILE ##
###################
global data
data = dict()
			
# open data file
if test == False:
	data['subject'] = vizinput.input('Subject ID?') 
	session = int(vizinput.input('Sitzung?')) 
	data['initials'] = vizinput.input('Initialien?')
	if session == 1:
		data['gender'] = vizinput.input('Geschlecht (m/f)?')
		data['age'] = int(vizinput.input('Alter?')) 		
else:
	data['subject'] = 1
	session = 1
	data['initials'] = 'test'

if session > 1:	
	pre_post = vizinput.input('Vor oder nach scan (pre/post)?') 
else:
	pre_post = 'pre'
	

saveDir = rootDir + 'datafiles/Subj_' + str(data['subject']) + '/session_' + str(session) + '/'	
stimDir = rootDir + 'stimuli/Subj_' + str(int(data['subject'])%100) + '/'

if not os.path.isdir(rootDir + 'datafiles/Subj_' + str(data['subject'])):
	os.mkdir(rootDir + 'datafiles/Subj_' + str(data['subject']))	
if not os.path.isdir(saveDir):
	os.mkdir(saveDir)
	
###check for existing data file <-uncomment for actual testing, along with input function above
fname = saveDir + 'data_'+str(data['subject'])+'_' + str(session) + '_' + pre_post + '_' +  data['initials'] + '_viz.txt'

# Do not overwrite data
if test == False:
	if os.path.isfile(fname):
		print('file already exists')
		viz.quit()

# Load data from session 1 if it exists
if session > 1:
	if test == True:
		load_from = rootDir + 'datafiles/Subj_' + str(data['subject']) + '/session_1/data_'+str(data['subject'])+'_1_pre' + '_' +  data['initials'] + '_viz.txt'
		with open(load_from) as json_data:
			data = json.load(json_data)	
	else:
		if session == 2 and pre_post == 'pre':
			load_from = rootDir + 'datafiles/Subj_' + str(data['subject']) + '/session_1/data_'+str(data['subject'])+'_1_pre' + '_' +  data['initials'] + '_viz.txt'
			print (load_from)
			if ( not os.path.isfile(fname) and os.path.isfile(load_from) ):
				print('Loading data file from session 1..........')
				with open(load_from) as json_data:
					data = json.load(json_data)
		elif session == 2 and pre_post == 'post':
			load_from = rootDir + 'datafiles/Subj_' + str(data['subject']) + '/session_2/data_'+str(data['subject'])+'_2_pre' + '_' +  data['initials'] + '_viz.txt'
			if ( not os.path.isfile(fname) and os.path.isfile(load_from) ):
				print('Loading data file from session 2 pre..........')
				with open(load_from) as json_data:
					data = json.load(json_data)	
		elif session == 3 and pre_post == 'pre':
			load_from = rootDir + 'datafiles/Subj_' + str(data['subject']) + '/session_2/data_'+str(data['subject'])+'_2_post' + '_' +  data['initials'] + '_viz.txt'
			print load_from
			if ( not os.path.isfile(fname) and os.path.isfile(load_from) ):
				print('Loading data file from session 2 post..........')
				with open(load_from) as json_data:
					data = json.load(json_data)		
		elif session == 3 and pre_post == 'post':
			load_from = rootDir + 'datafiles/Subj_' + str(data['subject']) + '/session_3/data_'+str(data['subject'])+'_3_pre' + '_' +  data['initials'] + '_viz.txt'
			if ( not os.path.isfile(fname) and os.path.isfile(load_from) ):
				print('Loading data file from session 3 pre..........')
				with open(load_from) as json_data:
					data = json.load(json_data)				
				
d = data
				
#elif session == 3:
#	if test == True:
#		with open(fname_sess2) as json_data:
#			d = json.load(json_data)	
#	else:
#		if ( not os.path.isfile(fname) and os.path.isfile(fname_sess2) ):
#			print('Loading data file from session 2..........')
#			with open(fname_sess2) as json_data:
#				d = json.load(json_data)


#	with open(stimDir+"block_order_subj_" + str(data['subject']) + ".dat") as f:
#		data['blockorder'] = [list(literal_eval(line)) for line in f]

###############
## VARIABLES ##
###############

global runNum
global count
global nruns
global hand_angle
global myProgressBar
global nTestRuns
global nPreTrainRun
global progressBarOn
global Message
global radius
global nTotalObjects
global nLearningTask
global prev_error
global error_lim
global nblocksValueTask
global displayFeedback
global freeExplore
global totalRew
global exploreTask

runNum = 0
ITI = 1
nobj = 12
nNewObj = 0
nRandObj = 0
nblocksValueTask = 12
totalRew = 0 #cumsum of reward over the entire experiment

nTotalObjects = nobj + nNewObj + nRandObj

nTotal = 16
nruns = 10	# number of exploration & positioning runs
nTestRuns = 2
nPreTrainRun = 2
radius = 15 #in meters, made global for convenience so be careful 
hand_angle = np.random.randint(0,360)

prev_error = [2*radius]
error_lim = 3

if pre_post == 'pre':
	data['session_' + str(session)] = dict()
data['session_' + str(session)][pre_post] = dict()
data['session_' + str(session)][pre_post]['freeExplore'] = dict()
data['session_' + str(session)][pre_post]['positionObject'] = dict()
data['session_' + str(session)][pre_post]['choiceTask'] = dict()

data['session_' + str(session)]['day'] = int(vizinput.input('Tag?')) 
	
#def initialiseDataStructure():
freeExplore = {}
freeExplore['orientation'], freeExplore['position'], freeExplore['time']    = [], [], []
	
positionObject = {}

def saveData():
	data['session_' + str(session)][pre_post]['freeExplore']['run_' + str(runNum)] = freeExplore.copy()		
	data['session_' + str(session)][pre_post]['positionObject']['run_' + str(runNum)] = positionObject.copy()		
	data['session_' + str(session)][pre_post]['choiceTask']['run_' + str(runNum)] = choiceTask.copy()		
	data['likert'] = likert.copy()
	with open(fname, 'w') as outfile:
		json.dump(data, outfile)

	#data['session_' + str(session)][pre_post]=data['session_' + str(session)]
	#with open(fname, 'w') as outfile:
	#	json.dump(data, outfile)

# Subject-specific data	
if session == 1:
	
	with open(rootDir+"stimuli/stimDistr.dat") as f:
		objPositions = [list(literal_eval(line)) for line in f]
	data['objPositions'] = objPositions	
	data['radius'] = radius
		
	with open(stimDir+"stimuli_subj_" + str(int(data['subject'])%100)+".dat") as f:
		stimuli = [list(literal_eval(line)) for line in f]
	data['stimuli'] = stimuli[0]
	
	nLearningTask = 10

elif session > 1:	# Load data from session 1
	
	objPositions = d['objPositions']
	radius = d['radius']
	stimuli = [d['stimuli']]
	
	data['objPositions'] = d['objPositions']
	data['radius'] = d['radius']
	data['stimuli'] = d['stimuli']
	
	nLearningTask = 1

##################
##MATH FUNCTIONS##
##################

def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x,y)
	
##############################
## ENVIRONMENT AND CONTROLS ##
##############################

# ground
ground = vizshape.addPlane(size=(40.0,40.0),axis=vizshape.AXIS_Y,scene=myScene2)
ground.setPosition(0,0,0)
t1 = viz.addTexture('images/tex2.jpg',wrap=viz.REPEAT)
ground.texture(t1)
ground.texmat( viz.Matrix.scale(20,20,1) )

# sky color
viz.clearcolor(0.5, 0.5, 0.5)
sky = viz.add(viz.ENVIRONMENT_MAP,'sky.jpg',scene=myScene2)
skybox = viz.add('skydome.dlc',scene=myScene2)
skybox.texture(sky)


#boundary elements
boundaries_root = viz.addGroup(scene=myScene2)
boundaries_root.visible(1)
data['boundary_pos'] = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
euler = [0, 90, -90, 180]
for b in range(4):
	boundary = viz.addChild('images/circle1.osg',parent=boundaries_root,pos=data['boundary_pos'][b],cache=viz.CACHE_CLONE,scene=myScene2)
	boundary.setEuler(euler[b], -90, 0)

##distal cues: add some trees
cues_root = viz.addGroup(scene=myScene2)
cues_root.visible(1)
if session == 1:
	cue_pos = []
	cue_identity = []
	
#	randomise which plant to position where
	v = np.random.permutation(7)
	for c in range(5):
		# distribute somewhat evenly
		cue_pos.append(72*c + np.random.randint(72));
		cue_identity.append(v[c])
	
else:
	cue_pos = d['cue_pos']
	cue_identity = d['cue_identity']


#for z in range(7):
#	s=0.5#scalefactor
#	cue = viz.addChild('images/Tree'+str(z)+'.osg',parent=cues_root,pos=[2,0,z-4],cache=viz.CACHE_CLONE)	
#	
#	if z==4 or z>5:				cue.setEuler([0,-90,0])
#	if z==1: 					cue.setScale([1.5*s,1.5*s,1.5*s])
#	if z==2: 					cue.setScale([0.5*s,0.5*s,0.5*s])
#	if z==5: 					cue.setScale([0.04*s,0.04*s,0.04*s])
#	if (z>2 and z!=5): 	cue.setScale([s,s,s])
#	if z==0: 					cue.setScale([0.9*s,0.9*s,0.9*s])
	
for c in range(5):
	x,y = pol2cart(radius + 4, cue_pos[c]*np.pi/180)
	z=cue_identity[c]
	s=2.0#scalefactor
	cue = viz.addChild('images/Tree'+str(z)+'.osg',parent=cues_root,pos=[x,0,y],cache=viz.CACHE_CLONE)	
	
	if z==4 or z>5:				cue.setEuler([0,-90,0])
	if z==1: 					cue.setScale([1.5*s,1.5*s,1.5*s])
	if z==2: 					cue.setScale([0.5*s,0.5*s,0.5*s])
	if z==5: 					cue.setScale([0.04*s,0.04*s,0.04*s])
	if (z>2 and z!=5): 			cue.setScale([s,s,s])
	if z==0: 					cue.setScale([0.9*s,0.9*s,0.9*s])

data['cue_pos'] = cue_pos
data['cue_identity'] = cue_identity
	
##lighting
mylight = viz.addLight() 
mylight.enable() 
mylight.position(0, 10, 0)
mylight.spread(180) ##uniform ambient lighting 
mylight.intensity(3.5)

##############################
########### OBJECTS ##########
##############################

# Object rotation. 
# The speed of the timer.  A value of 0 means the timer function
# will be called every frame
UPDATE_RATE = 0
#A variable that will hold the angle
angle = 0
#The speed of the rotation
SPEED = 30

def rotateModel():
	global angle

	#Increment the angle by the rotation speed based on elapsed time
	angle = angle + (SPEED * viz.elapsed())

	#Update the models rotation
	for o in range(nobj): 
		plants[o].setEuler([angle,270,0])

global popup_obj
popup_obj = False

#fade to visible when viewpoint moves near and popup_obj == True
def EnterSphere(e, sphere):
	global popup_obj
	if popup_obj == True:
		sphere.visible(1)
		for x in range(nNewObj):
			plants[nobj + x].visible(0)
	
#fade to invisible when viewpoint moves away
def ExitSphere(e, sphere):
	global exploreTask
	if exploreTask:
		sphere.visible(0)	

#add spheres and create a proximity sensor around each one
sphereSensors, plants = [], []
plants_root = viz.addGroup(scene=myScene2)
plants_root.visible(1)
count = 0

for position in objPositions:
	pos_rad = [x * radius for x in position]
	
	plant = viz.addChild('.\images\obj%02d' % data['stimuli'][count] + '.osg',parent=plants_root,pos=pos_rad,cache=viz.CACHE_CLONE,scene=myScene2)
		
	plant.setAxisAngle([0, 0, 0 , np.random.randint(360)]) 
	plant.setScale([0.1,0.1,0.1])
	plant.visible(0)
	
	sensor = vizproximity.addBoundingSphereSensor(plant,scale=5)
	sensor.name = 'obj%02d' % count
	sphereSensors.append(sensor)
	
	manager.addSensor(sensor)
	manager.onEnter(sensor, EnterSphere, plant)
	manager.onExit(sensor, ExitSphere, plant)
	
	plants.append(plant)
	count += 1
	
	
# Object rotation
#setup a timer and specify it's rate and the function to call
vizact.ontimer(UPDATE_RATE, rotateModel)

#setup keyboard controls
vizcam.KeyboardCamera(forward=viz.KEY_UP,
						backward=viz.KEY_DOWN,
						left=None,
						right=None,
						up='d' ,
						down=None,
						turnRight=viz.KEY_RIGHT,
						turnLeft=viz.KEY_LEFT,
						pitchDown=None ,
						pitchUp='g' ,
						rollRight=None ,
						rollLeft=None ,
						moveMode=viz.REL_LOCAL,
						moveScale=1.0,
						turnScale=0.5)

#turn on collisions
viz.collision(viz.ON)

##################
## INSTRUCTIONS ##
##################

def Instructions(messageType):
	global Message
	global displayFeedback
	global quad
	
	viz.MainView.setScene(myScene1)
	if messageType == 'GeneralInstrSession1': 
		Message = vizinfo.InfoPanel('Herzlich Willkommen zu Sitzung ' + str(session) + ' und vielen Dank für Ihre Teilnahme an dieser Studie!\n\n' +
		'Dieses Experiment besteht aus mehreren Teilen. Vor jedem Teil bekommen Sie\n\ngenaue Instruktionen. ' +
		'Falls trotzdem etwas unklar sein sollte, sprechen Sie uns gern jederzeit an.\n\n' +
		'Beginnen Sie mit ENTER.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'GeneralInstrSession2': 
		Message = vizinfo.InfoPanel('Willkommen zurück!\n\n' + 
		"Zu Beginn der heutigen Sitzung haben Sie nochmal die Möglichkeit sich in der Arena umzuschauen,\n\n" +
		"damit Sie sich die Positionen der Monster wieder in Erinnerung rufen können.\n\n" +
		"Anschließend werden wir Sie nochmals bitten die Monster an ihrer Position zu platzieren.\n\n" +
		"Nutzen Sie diesen Teil um sich die Positionen der Monster nochmals genau einzuprägen.\n\n" +
		"Je besser Sie sich an die Positionen erinnern, desto mehr Punkte können Sie in einem späteren Teil der Studie gewinnen.\n\n" +
		'Sprechen Sie uns gern jederzeit an, falls etwas unklar ist.\n\n' +
		'Beginnen Sie mit ENTER.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'GeneralInstrSession3': 
		Message = vizinfo.InfoPanel('Wir bitten Sie die Monster noch ein Mal zu positionieren.\n\n' + 
		'Wundern Sie sich nicht: dieses Mal bekommen Sie kein Feedback.\n\n' + 
		'Beginnen Sie mit ENTER.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'presentObjectsInstruct':
		Message = vizinfo.InfoPanel('Wir werden Ihnen nun einige Monster präsentieren.\n\n' + 
		'Machen Sie sich mit den Monstern vertraut, diese werden in allen Teilen des Experiments wichtig sein.\n\n' + 
		'Sie können diesen Teil beginnen und durch die Monster klicken, indem Sie ENTER drücken.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)      
	elif messageType == 'freeExplorationInstruct':
		Message = vizinfo.InfoPanel('Im nächsten Teil des Experiments werden Sie sich in einer Arena befinden. Mit den Pfeiltasten können Sie durch die Arena navigieren.\n\n' +
		'An bestimmten Orten werden Monster auftauchen. Jedes Monster gehört an eine ganz bestimmte Position. Versuchen Sie sich \n\n' +
		'diese Position genau einzuprägen. Je besser Sie sich an die Position der Monster erinnern können, desto mehr Punkte können Sie \n\n' + 
		'in einem späteren Teil der Studie gewinnen.\n\n' +
		'Sie haben so viel Zeit wie Sie möchten, um sich in der Arena umzuschauen. Wenn Sie sich alle Positionen der Monster gemerkt haben,\n\n' + 
		'drücken Sie ENTER, um zum nächsten Teil des Experiments überzugehen.\n\n' +
		'Haben Sie noch Fragen? Falls ja, wenden Sie sich bitte an den Experimenter.\n\n' +
		'Beginnen Sie mit ENTER. \n\n' ,align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'positionObject_instruct':
		s1 = ('Wir möchten Sie nun darum bitten, die Monster an den Ort zu platzieren, an den sie gehören.\n\n' + 
		'Zunächst werden wir Ihnen ein Monster zeigen. Merken Sie es sich gut. Wenn Sie ENTER drücken,\n\n' + 
		'werden Sie sich in der Arena wiederfinden. Navigieren Sie zu dem Ort, an den das Monster gehört\n\n' + 
		'und drücken Sie ENTER wenn Sie angekommen sind.\n\n') 
		if displayFeedback == True:
			s2 = ('Ein smiley (siehe unten) wird Ihnen anzeigen, wie weit Sie von der korrekten Position entfernt sind.\n\n' +
			'Sie können sich dann auch in der Arena umsehen, um festzustellen, wo das Monster tatsächlich hingehört.\n\n' + 
			'Laufen Sie zu der richtigen Position und drücken Sie ENTER, um das Monster aufzulesen. Versuchen Sie\n\n' + 
			'durch dieses Feedback zu lernen, damit Sie das Monster beim nächsten Mal noch akkurater positionieren können.\n\n' + 
			'Sie werden bis zu zehn Runden durchlaufen, abhängig davon, wie schnell Sie die Positionen der Monster lernen.\n\n')
			
			#Create texture from image file
			image = viz.add('./images/allsmileys.png')

			#Add quad to screen
			#Display image in lower left corner of screen
			quad = viz.addTexQuad(parent=viz.ORTHO)
			quad.setPosition([screensize[0]/2, screensize[1]/3, 50]) #put quad in view 

			quad.alignment(viz.ALIGN_CENTER)
			quad.texture(image)
			quad.setScale(image.getSize())
			
			#Wait for next frame to be drawn to screen
			d_on = yield viztask.waitDraw()						
		else: 
			s2 = ('Heute bekommen Sie kein Feedback.\n\n')
		s3 = ('Haben Sie noch Fragen? Falls ja, wenden Sie sich bitte jetzt an den Experimenter.\n\n' +
		'Beginnen Sie mit ENTER.')
		Message = vizinfo.InfoPanel(s1+s2+s3,align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'valueTaskInstructions1':
		Message = vizinfo.InfoPanel('Dieser Teil des Experiments besteht aus mehreren Blöcken.\n\n' +
		'In jedem Block müssen Sie eine Entscheidung zwischen je zwei Monstern treffen.\n\n' + 
		'Each monster has a value, and by choosing it, you will earn the corresponding number of points.\n\n' +
		"After making a choice the chosen monster's value will be shown to you below the monster.\n\n" +
		'You can make the choice using the left and the right arrows.\n\n' +
		'Press ENTER to see an example.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'valueTaskInstructions2':
		Message = vizinfo.InfoPanel('How valuable an monster is depends on its location in space.\n\n' + 
		'Monsters that are located near each other in the arena also have a similar value.\n\n' + 
		'This means that you can infer the value of monsters based on what you know about the value of other monsters.\n\nPress ENTER to proceed.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'valueTaskInstructions3':
		Message = vizinfo.InfoPanel('Try to estimate how many points you can get for each of the two monsters as well as you can\n\n' + 
		'based on everything you know about the value of some of the monsters and how far they are in the arena from each other.\n\n' +
		'A good estimation will pay off: You will gain the number of points corresponding to the value of the monster you choose.\n\n' + 
		'At the end of the study, the points you have collected will be converted in cash and paid to you on top of the baseline fee (10 points = 1 NOK).\n\n' + 
		'ress ENTER to proceed',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'valueTaskInstructions4':
		Message = vizinfo.InfoPanel('You will notice that there are a few monsters whose location in the arena you have not learned.\n\n' +
		'Try to do your best for these monsters. Over time you may figure out where they belong.\n\n',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'valueTaskInstructions5':
		Message = vizinfo.InfoPanel('IMPORTANT:\n\nWithin each block the values of each monster are stay the same. From one block to the next, the values will change.\n\n' +
		'However, it is always the case that monsters that are nearby in the arena also have a similar value.\n\n' +
		'At the end of a block, you will be asked to position some monsters in the arena again.\n\n' + 		
		'Any questions? Ask the experimenter.\n\n' +
		'Start with ENTER. \n\n',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'valueFeedback':
		Message = vizinfo.InfoPanel('Well done!\n\nYou collected\n\n' +
		str(int(choiceTask['totalRew']*10)) + ' points in this block.\n\n' + 
		'Proceed with ENTER. \n\n',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'feedback_posObject':
		Message = vizinfo.InfoPanel('Ende des Blocks\n\nDurchschnittlicher Fehler:\n' + str(np.round(np.mean(prev_error)*10)/10) + 'm\n\nDrücken Sie ENTER um fortzufahren.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'goodJob': 
		Message = vizinfo.InfoPanel('Sehr gut!\n\nDrücken Sie ENTER um fortzufahren.',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'globalEnd_session2':
		Message = vizinfo.InfoPanel('ENDE\n\nVielen Dank für Ihre Teilnahme an unserem Experiment!',align=viz.ALIGN_CENTER_TOP,key=None,icon=False)   
	elif messageType == 'globalEnd_session2':
		Message = vizinfo.InfoPanel('ENDE\n\n'+ 
		'Sie haben ' + int(totalRew*10) + ' Punkte in diesem Experiment gesammelt.\n\n' +
		'Das entspricht einer Bonuszahlung von ' + int(math.ceil(totalRew)) + ' Euro\n\n' +
		'Vielen Dank für Ihre Teilnahme an unserem Experiment!',align=viz.ALIGN_CENTER_TOP,key=None,icon=False
		)
	
	#Message.bgcolor([0.5, 0.5, 0.5])
	Message.visible(viz.ON)

#####################################
####### FREE EXPLORATION TASK  ######
#####################################
	
def freeExploration():
	global exploreTask
	exploreTask = True
	viz.MainView.setScene(myScene2)
	freeExplore['start'] = time.clock()
	
	#teleport to new location each trial
	x , y = pol2cart(np.random.randint(2,radius-2), np.random.randint(0,360))
	
	viz.MainView.setPosition(x,0,y) 
	viz.MainView.setEuler(np.random.randint(0,360),0,0)
	
	freeExplore['position'] = [x,0,y]
	freeExplore['time'] = [time.clock()]
	freeExplore['orientation'] = [viz.MainView.getEuler()]
	
	explore_timer = vizact.ontimer(0.5, getPositionData_freeExplore)	
	yield viztask.waitKeyDown(viz.KEY_RETURN)
	explore_timer.setEnabled(0)
	freeExplore['end'] = time.clock()
	
def getPositionData_freeExplore():
	freeExplore['orientation'].append(viz.MainView.getEuler())
	freeExplore['position'].append(viz.MainView.getPosition())
	freeExplore['time'].append(time.clock())

def getPositionData_positionObject():
	global c
	positionObject['trial_' + str(c)]['orientation'].append(viz.MainView.getEuler())
	positionObject['trial_' + str(c)]['position'].append(viz.MainView.getPosition())
	positionObject['trial_' + str(c)]['time'].append(time.clock())
	
def getPositionData_positionObject_Feedback():
	global c
	positionObject['trial_' + str(c)]['feedback']['orientation'].append(viz.MainView.getEuler())
	positionObject['trial_' + str(c)]['feedback']['position'].append(viz.MainView.getPosition())
	positionObject['trial_' + str(c)]['feedback']['time'].append(time.clock())
	
def getSliderValues():
	global c
	likert['trial_' + str(c)]['position'].append(mySlider.get())
	likert['trial_' + str(c)]['time'].append(time.clock())
	
def presentFixationCross(iti):
		
	# Fixation cross
	image = viz.add(rootDir + 'images/fixation.png')
	quad = viz.addTexQuad(parent=viz.ORTHO)
	quad.setPosition([screensize[0]/2, screensize[1]/2, 100]) #put quad in view 

	quad.alignment(viz.ALIGN_CENTER)
	quad.texture(image)
	quad.setScale(image.getSize())
	yield viztask.waitDraw()
	yield viztask.waitTime(iti)
	
	quad.remove()
	

#####################################
########## VALUE TASK  ##############
#####################################

# Create arrays to store data
choiceTask = {}
choiceTask['RT'], choiceTask['decision'], choiceTask['cr'], choiceTask['chosen_value'], choiceTask['unchosen_value'], choiceTask['start'], choiceTask['end'], choiceTask['block']   = [], [], [], [], [], [], [], []
				
def valueTask():
	global progressBarOn
	global myProgressBar
	global totalRew
	global quad0
	global quad1
	
	viz.MainWindow.setScene(2)
	choiceTask['start'] = time.clock()
	choiceTask['block'] = data['blockorder'][0][runNum-1]
	
	for o in range(2):
		print(o)
		with open(stimDir+"choice_option" + str(o) + "_subj_" + str(data['subject'])+"_block_" + str(choiceTask['block']) + ".dat") as f:
			choiceTask['choice_option' + str(o)] = [list(literal_eval(line)) for line in f]
		with open(stimDir+"choice_value" + str(o) + "_subj_" + str(data['subject'])+"_block_" + str(choiceTask['block']) + ".dat") as f:
			choiceTask['choice_value' + str(o)] = [list(literal_eval(line)) for line in f]
	
	# Store choice data
	choiceTask['decision'] = []
	choiceTask['chosen_value'] = []
	choiceTask['unchosen_value'] = []
	choiceTask['cr'] = []
	choiceTask['RT'] = []
	
	for c in range(len(choiceTask['choice_option0'][0])):	
		
		choice = yield choiceTrial([data['stimuli'][choiceTask['choice_option0'][0][c]], data['stimuli'][choiceTask['choice_option1'][0][c]]])			
		choiceTask['RT'].append(choice[1].time - choice[0].time)
	
		if choice[1].key == viz.KEY_LEFT:
			choiceTask['decision'].append(0)
			choiceTask['chosen_value'].append(choiceTask['choice_value0'][0][c])
			choiceTask['unchosen_value'].append(choiceTask['choice_value1'][0][c])
			
			if choiceTask['choice_value0'][0][c] > choiceTask['choice_value1'][0][c]:
				choiceTask['cr'].append(1)					
			else:
				choiceTask['cr'].append(0)
			quad1.remove()
			pos = 0.3
		else: 
			choiceTask['decision'].append(1)
			choiceTask['chosen_value'].append(choiceTask['choice_value1'][0][c])
			choiceTask['unchosen_value'].append(choiceTask['choice_value0'][0][c])
			
			if choiceTask['choice_value1'][0][c] > choiceTask['choice_value0'][0][c]:
				choiceTask['cr'].append(1)					
			else:
				choiceTask['cr'].append(0)
			quad0.remove()
			pos = 0.7
				
		progressBarOn = True
		presentProgressBar(choiceTask['chosen_value'][c],pos)
		yield viztask.waitTime(2)
		
		quad0.remove()
		quad1.remove()
		myProgressBar.remove()
		yield presentFixationCross(0.5)
		
		saveData()
		
	choiceTask['totalRew'] = np.sum(choiceTask['chosen_value'])
	totalRew += choiceTask['totalRew']
	
	choiceTask['end'] = time.clock()
	
	
#####################################
###### Object positioning task ######
#####################################

def positionObjectTask(n):
	global displayFeedback
	global prev_error
	global c
	global popup_obj
	global exploreTask
	exploreTask = False
	viz.MainView.setScene(myScene1)
	
	positionObject['start'] = time.clock()
	stimOrder = np.random.permutation(n)
	
	# Do not present the objects
	for c in range(12):
		plants[c].visible(0)

	for c in range(n):	
		viz.MainView.setScene(myScene1)
	
		popup_obj = False
		collectMessage = 'Finde dieses Monster:'
		
		#info
		info = vizinfo.InfoPanel(collectMessage,align=viz.ALIGN_CENTER_TOP,key=None,icon=False) 
		#info.fontSize(36)
		info.color(viz.BLACK)
		#info.setPosition([0.4,0.8,0])
		#info.visible(1)
		
		# Create texture from image file
		image = viz.add('.\images\obj%02d' % data['stimuli'][stimOrder[c]] + '.png')
	
		#Add quad to screen
		# Display image in lower left corner of screen
		quad = viz.addTexQuad(parent=viz.ORTHO)
		quad.setPosition([screensize[0]/2, screensize[1]/2, 10]) #put quad in view 

		#Wait for next frame to be drawn to screen
		d = yield viztask.waitDraw()
	
		#Save display time
		displayTime = d.time
	
		quad.alignment(viz.ALIGN_CENTER)
		quad.texture(image)
		quad.setScale(image.getSize())

		#Wait for keyboard reaction
		d = yield viztask.waitKeyDown(viz.KEY_RETURN)
		
		viz.MainView.setScene(myScene2)
	
		#Calculate reaction time
		positionObject['trial_' + str(c)] = dict()		
		
		quad.remove()
		info.visible(0)
	
		#teleport to new location each trial
		x, y = pol2cart(np.random.randint(2,radius-2), np.random.randint(0,360))
		viz.MainView.setPosition(x,0,y) 
		viz.MainView.setEuler(np.random.randint(0,360),0,0)
	
		positionObject['trial_' + str(c)]['whichPNG'] = data['stimuli'][stimOrder[c]]
		positionObject['trial_' + str(c)]['stim'] = stimOrder[c]
		positionObject['trial_' + str(c)]['RT'] = d.time - displayTime
		positionObject['trial_' + str(c)]['position'] = [x,0,y]
		positionObject['trial_' + str(c)]['time'] = [time.clock()]
		positionObject['trial_' + str(c)]['orientation'] = [viz.MainView.getEuler()]	
	
		#Add the object that will do the grabbing
		hand = viz.add('.\images\location.png')
		quad2 = viz.addTexQuad(parent=viz.ORTHO)
		quad2.setPosition([screensize[0]/2, screensize[1]/12, 100]) #put quad in view 
		quad2.alignment(viz.ALIGN_CENTER)
		quad2.texture(hand)
		quad2.setScale([33,55,1])

		position_timer = vizact.ontimer(0.5, getPositionData_positionObject)
		yield viztask.waitKeyDown(viz.KEY_RETURN)
		position_timer.setEnabled(0)
		
		hand_pos = viz.MainWindow.screenToWorld(quad2.getPosition())
		
		if stimOrder[c] < 14:
			pos_target = [x * radius for x in data['objPositions'][stimOrder[c]]]
			positionObject['trial_' + str(c)]['hand'] = hand_pos
			hand_pos[1] = 0
			positionObject['trial_' + str(c)]['target_location'] = pos_target
			positionObject['trial_' + str(c)]['error'] = vizmat.Distance(hand_pos,pos_target)
			
			if displayFeedback == True:
				plants[stimOrder[c]].visible(1)
		
		positionObject['trial_' + str(c)]['drop_time'] = d.time
		quad2.remove()
		
		# Create texture from image file
		if displayFeedback == True:
			
			threshold = [2,4,6,8]
			if positionObject['trial_' + str(c)]['error'] < threshold[0]:
				feedback = '1'
			elif positionObject['trial_' + str(c)]['error'] > threshold[0] and positionObject['trial_' + str(c)]['error'] < threshold[1]:
				feedback = '2'
			elif positionObject['trial_' + str(c)]['error'] > threshold[1] and positionObject['trial_' + str(c)]['error'] < threshold[2]:
				feedback = '3'
			elif positionObject['trial_' + str(c)]['error'] > threshold[2] and positionObject['trial_' + str(c)]['error'] < threshold[3]:
				feedback = '4'
			elif positionObject['trial_' + str(c)]['error'] > threshold[3]:
				feedback = '5'
			
			image = viz.add('./images/smiley_'+feedback+'.png')

			#Add quad to screen
			quad = viz.addTexQuad(parent=viz.ORTHO)
			quad.setPosition([screensize[0]/2, 2*screensize[1]/3, 100]) #put quad in view 
			#hand.visible(viz.OFF)
		
			#Wait for next frame to be drawn to screen
			d = yield viztask.waitDraw()
		
			#Save display time
			positionObject['trial_' + str(c)]['feedback'] = dict()
			positionObject['trial_' + str(c)]['feedback']['position'] = []
			positionObject['trial_' + str(c)]['feedback']['time'] = []
			positionObject['trial_' + str(c)]['feedback']['orientation'] = []
		
			quad.alignment(viz.ALIGN_CENTER)
			quad.texture(image)
			quad.setScale(image.getSize())
			
			position_timer = vizact.ontimer(0.5, getPositionData_positionObject_Feedback)
			
			plants[stimOrder[c]].visible(1)
			
			d = yield viztask.waitKeyDown(viz.KEY_RETURN)
			while vizmat.Distance(viz.MainView.getPosition(),pos_target) > 3:
				print(vizmat.Distance(viz.MainView.getPosition(),pos_target))
				d = yield viztask.waitKeyDown(viz.KEY_RETURN)
			#popup_obj = True
			
			position_timer.setEnabled(0)		
			
			quad.remove()
			objectPres['smiley_off_time'].append(d.time)
		
		saveData()
		info.visible(0)
		
		if stimOrder[c] < 14:
			plants[stimOrder[c]].visible(0)
			prev_error.append(np.mean(positionObject['trial_' + str(c)]['error']))
			
		print prev_error
		
	positionObject['end'] = time.clock()
	
#####################################
######## OBJECT PRESENTATION ########
#####################################

# Create arrays to store data
objectPres = {}
objectPres['RT'], objectPres['obj'], objectPres['stim'], objectPres['orientation'], objectPres['position'], objectPres['smiley_off_time'], objectPres['drop_location'], objectPres['target_location'], objectPres['drop_time'], objectPres['error'] = [], [], [], [], [], [], [], [], [], []

def presentObjects(whichObj, value):
	global quad
	global progressBarOn
	
	viz.MainView.setScene(myScene1)
	yield presentFixationCross(0.5)

	#Create texture from image file
	image = viz.add('.\images\obj%02d' % whichObj + '.png')

	#Add quad to screen
	#Display image in lower left corner of screen
	quad = viz.addTexQuad(parent=viz.ORTHO)
	quad.setPosition([screensize[0]/2, screensize[1]/2, 100]) #put quad in view 

	yield presentProgressBar(value,0.5)

	#Wait for next frame to be drawn to screen
	d = yield viztask.waitDraw()
	
	quad.alignment(viz.ALIGN_CENTER)
	quad.texture(image)
	quad.setScale(image.getSize())
	
	#Return display time
	viztask.returnValue(d.time)
	
def choiceTrial(whichObjs):	
	global quad0
	global quad1
	global wo
	viz.MainView.setScene(myScene2)
	
	#Create texture from image file
	image1 = viz.add('.\images\obj%02d' % whichObjs[0] + '.png')
	image2 = viz.add('.\images\obj%02d' % whichObjs[1] + '.png')

	#Add quad to screen
	#Display image in lower left corner of screen
	quad0 = viz.addTexQuad(parent=viz.ORTHO)
	quad0.setPosition([screensize[0]/3, screensize[1]/2, 100]) #put quad in view 

	quad1 = viz.addTexQuad(parent=viz.ORTHO)
	quad1.setPosition([2*screensize[0]/3, screensize[1]/2, 100]) #put quad in view 

	quad0.alignment(viz.ALIGN_CENTER)
	quad0.texture(image1)
	quad0.setScale(image1.getSize())
	
	quad1.alignment(viz.ALIGN_CENTER)
	quad1.texture(image2)
	quad1.setScale(image2.getSize())
	
	#Wait for next frame to be drawn to screen
	d_on = yield viztask.waitDraw()		
	d_off = yield viztask.waitKeyDown([viz.KEY_LEFT, viz.KEY_RIGHT])
	
	#Return display time
	viztask.returnValue([d_on, d_off])
	
def presentProgressBar(value, pos):
	global progressBarOn
	global myProgressBar
	
	myProgressBar = viz.addProgressBar('',scene = 2)
	myProgressBar.set(value)
	myProgressBar.setPosition(([pos,.2,0]))
	
	if progressBarOn==False:
		myProgressBar.visible(viz.OFF)	
		
def onMouseDown(button): 
	global myProgressBar
	if button == viz.MOUSEBUTTON_LEFT:
		yield myProgressBar.get()	

##############################
#!! MAIN EXPERIMENTAL LOOP !!#
##############################

# Create arrays to store data
likert = {}

def EXPERIMENT(ITI):
	global popup_obj
	global runNum
	global myProgressBar
	global progressBarOn
	global Message
	global displayFeedback
	global prev_error
	global sliderValue
	global quad
	global mySlider
	global c
		
	#Proceed through experiment phases
	data['start_time'] = viz.tick()	
		
	# General instructions
	#yield Instructions('GeneralInstrSession' + str(session))
	#yield viztask.waitKeyDown([viz.KEY_RETURN])
	#Message.visible(viz.OFF)
	
	try:
		if session == 1:
			
			displayFeedback = True
				
			##############################
			#### OBJECT PRESENTATION #####
			##############################	
			
			yield Instructions('presentObjectsInstruct')
			yield viztask.waitKeyDown([viz.KEY_RETURN])
			Message.visible(viz.OFF)			
					
			if test == False:
				stimPres = np.random.permutation(nTotalObjects)
				for c in range(nTotalObjects):
					
					progressBarOn=False
					displayTime = yield presentObjects(data['stimuli'][stimPres[c]],0)
						
					#Wait for keyboard reaction
					d = yield viztask.waitKeyDown(viz.KEY_RETURN)
					quad.remove()
					
					objectPres['obj'].append(stimPres[c])
					objectPres['stim'].append(data['stimuli'][stimPres[c]])
					objectPres['RT'].append(d.time - displayTime)

					data['session_' + str(session)]['objectPres'] = objectPres
					
					saveData()
					#Calculate reaction time
			
			##############################
			####### LEARNING TASK ########
			##############################	
			
			# Cycle through the experimental runs and create arrays to store data
			for r in range(nruns):
				runNum = r+1
					
				data['session_' + str(session)][pre_post]['freeExplore']['run_' + str(runNum)] = dict()
				data['session_' + str(session)][pre_post]['positionObject']['run_' + str(runNum)] = dict()
					
				if (np.round(np.mean(prev_error)*10)/10) > error_lim or r < 5:
					
					prev_error = []
					Message = vizinfo.InfoPanel('Runde ' + str(r+1) + '\n\n' + 
					'Beginnen Sie mit ENTER.',align=viz.ALIGN_CENTER_TOP, icon=False)
					Message.visible(viz.ON)
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
				
					yield Instructions('freeExplorationInstruct')
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
					
					popup_obj = True
					yield freeExploration()
					yield Instructions('goodJob')
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
				
					yield Instructions('positionObject_instruct')
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
					quad.remove()
					
					popup_obj = False
					displayFeedback = True
					yield positionObjectTask(nobj)
					yield Instructions('feedback_posObject')
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
				
					saveData()
					
				else:
					viz.MainView.setScene(myScene1)
						
					Message = vizinfo.InfoPanel('Sie haben diese Sitzung erfolgreich beendet. Vielen Dank für Ihre Teilnahme!',align=viz.ALIGN_CENTER_TOP, icon=False)
					Message.visible(viz.ON)
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
									
					
		elif session > 1:
			if session == 2 and pre_post=='pre':
				displayFeedback = True	
				yield Instructions('GeneralInstrSession2')		
			else:
				displayFeedback = False
				yield Instructions('GeneralInstrSession3')

			yield viztask.waitKeyDown([viz.KEY_RETURN])
			Message.visible(viz.OFF)
			
			saveData()
			
			##############################
			####### REMINDER TASK ########
			##############################	

			# Cycle through the experimental runs and create arrays to store data	
			for r in range(1):
				runNum = r
				
				if session == 2 and pre_post=='pre':
					yield Instructions('freeExplorationInstruct')
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					Message.visible(viz.OFF)
				
					data['session_' + str(session)][pre_post]['freeExplore']['run_' + str(runNum)] = dict()
					data['session_' + str(session)][pre_post]['positionObject']['run_' + str(runNum)] = dict()
				
					# Exploration task
					popup_obj = True
					yield freeExploration()
			
					yield Instructions('goodJob')
					yield viztask.waitKeyDown([viz.KEY_RETURN])
					
					Message.visible(viz.OFF)
			
				yield Instructions('positionObject_instruct')
				yield viztask.waitKeyDown([viz.KEY_RETURN])
				Message.visible(viz.OFF)
				if displayFeedback == True:
					quad.remove()
				
				popup_obj = False
				yield positionObjectTask(nobj)
				yield Instructions('feedback_posObject')
				yield viztask.waitKeyDown([viz.KEY_RETURN])
				Message.visible(viz.OFF)
			
				saveData()

			##############################
			### INSTRUCTIONS SESSION 2 ###
			##############################

		data['end_time'] = viz.tick()
	
	finally:
		saveData()
	
	yield Instructions('globalEnd_session' + str(session))
	yield viztask.waitKeyDown([viz.KEY_RETURN])
	viz.quit()

##################################
##WRITE OUT PATH DATA FUNCTION####
##################################

##LAUNCH EXPERIMENT####
global startTime
startTime = time.clock()
viztask.schedule(EXPERIMENT(ITI))
