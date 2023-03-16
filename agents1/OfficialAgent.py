import sys, random, enum, ast, time, csv
from datetime import datetime
from matrx import grid_world
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

class Phase(enum.Enum):
    INTRO = 1,
    FIND_NEXT_GOAL = 2,
    PICK_UNSEARCHED_ROOM = 3,
    PLAN_PATH_TO_ROOM = 4,
    FOLLOW_PATH_TO_ROOM = 5,
    PLAN_ROOM_SEARCH_PATH = 6,
    FOLLOW_ROOM_SEARCH_PATH = 7,
    PLAN_PATH_TO_VICTIM = 8,
    FOLLOW_PATH_TO_VICTIM = 9,
    TAKE_VICTIM = 10,
    PLAN_PATH_TO_DROPPOINT = 11,
    FOLLOW_PATH_TO_DROPPOINT = 12,
    DROP_VICTIM = 13,
    WAIT_FOR_HUMAN = 14,
    WAIT_AT_ZONE = 15,
    FIX_ORDER_GRAB = 16,
    FIX_ORDER_DROP = 17,
    REMOVE_OBSTACLE_IF_NEEDED = 18,
    ENTER_ROOM = 19

class TrustBelief:
    path = '/beliefs/allTrustBeliefs.csv'
    default = 1.0
    basicChange = 0.5
    attributes = ['competence', 'willingness']
    def __init__(self, humanName, folder) -> None:
        self.folder = folder
        self.competence = 1.0
        self.willingness = 1.0
        self.humanName = humanName

        trustfile_header = []
        trustfile_contents = []
        with open(folder + TrustBelief.path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if trustfile_header==[]:
                    trustfile_header=row
                    continue
                # Retrieve trust values
                if row and row[0]==humanName:
                    self.competence = float(row[1])
                    self.willingness = float(row[2])

                # Initialize default trust values
                if row and row[0]!=self.humanName:
                    self.competence = TrustBelief.default
                    self.willingness = TrustBelief.default
    
    def get_binary_willingness(self):
        willingness = (self.willingness + 1) / 2
        return np.random.choice([0, 1], 1, p=[1-willingness, willingness])
    
    
    def get_binary_competence(self):
        competence = (self.competence + 1) / 2
        return np.random.choice([0, 1], 1, p=[1-competence, competence])
    
    def get_trust(self):
        competence = (self.competence + 1) / 2
        willingness = (self.willingness + 1) / 2
        trust = (competence + willingness) / 2
        return np.random.choice([0, 1], 1, p=[1-trust, trust])

    def updateCompetence(self, percent, withFlush = False):
        self.competence = np.clip(self.competence + TrustBelief.basicChange * percent, -1, 1)
        if withFlush:
            self.flushUpdates()
    
    def updateWillingness(self, percent, withFlush = False):
        self.willingness = np.clip(self.willingness + TrustBelief.basicChange * percent, -1, 1)
        if withFlush:
            self.flushUpdates()

    def flushUpdates(self):
        with open(self.folder + TrustBelief.path, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name','competence','willingness'])
            csv_writer.writerow([self.humanName,self.competence,self.willingness])


class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._slowdown = slowdown
        self._condition = condition     # Condition of the human
        self._humanName = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []         # found and saved
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._sendMessages = []
        self._currentDoor = None
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._goalVic = None
        self._goalLoc = None
        self._humanLoc = None
        self._distanceHuman = None
        self._distanceDrop = None
        self._agentLoc = None
        self._todo = []                 # Victims found but not saved
        self._answered = False
        self._tosearch = []
        self._carrying = False
        self._waiting = False
        self._rescue = None
        self._recentVic = None
        self._receivedMessages = []
        self._moving = False
        self._overrideRescueAlone = False
        self._nr_skipped_action = 0    # count for number of actions that are skipped in a row because of a lack of trust
        self._max_allowed_skips = 7     # number of skipping some actions in a row that is allowed, if this is reached, then skipping is not allowed and this value is resetted
        self._searched_rooms_by_robot = []
        # self._time_started_waiting_for_help = datetime.datetime.now() + datetime.timedelta(weeks=52)    # arbitrary high value as waiting hasn't started yet
        # self._max_time_difference = 3   # max time allowed for waiting
        self._overrideContinue = False
        self._overrideRemoveAlone = False
        self._timeStartedWaiting = datetime.now()
        self.waitingForDecisionResponse = False
        self._max_wait_time = 30

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action
        return state
        
    print_counter = 0
    def decide_on_actions(self, state):
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        
        
        # print(BaselineAgent.aux)
        # Create a list of received messages from the human team member
        for mssg in self.received_messages:
            for member in self._teamMembers:
                 # as the decide_on_action is called in a loop, mssg.content might not change and we don't want to add the same message over and over again
                if mssg.from_id == member and mssg.content not in self._receivedMessages: 
                    self._receivedMessages.append(mssg.content)
                # elif mssg.from_id == member:
                #     print(mssg.content)
                #     print(self._receivedMessages)
        # Process messages from team members
        # self._processMessages(state, self._teamMembers, self._condition)
        # Initialize and update trust beliefs for team members
        # trustBeliefs = self._loadBelief(self._teamMembers, self._folder)
        trustBeliefs = TrustBelief(self._humanName, self._folder)
        self._processMessages(state, self._teamMembers, self._condition, trustBeliefs)
        self._trustBelief(self._teamMembers, trustBeliefs, self._folder, self._receivedMessages)
        
        if BaselineAgent.print_counter % 30 == 0: # printing every 30 iterations how the competence and willingness are evolving
            print('competence ' + str(trustBeliefs.competence))
            print('willingness ' + str(trustBeliefs.willingness))
        BaselineAgent.print_counter += 1

        # reset messages to no new ones after processing them
        self.received_messages = []
        self._receivedMessages = []

        # Check whether human is close in distance
        if state[{'is_human_agent': True}]:
            self._distanceHuman = 'close'
        if not state[{'is_human_agent': True}]:
            # Define distance between human and agent based on last known area locations
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'far'
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'close'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'far'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'close'

        # Define distance to drop zone based on last known area location
        if self._agentLoc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3, 4, 7, 10, 13, 14]:
            self._distanceDrop = 'close'

        # Check whether victims are currently being carried together by human and agent
        for info in state.values():
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id'] or \
                'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying']) > 0 and 'mild' in info['is_carrying'][0]['obj_id'] and self._rescue=='together' and not self._moving:
                # If victim is being carried, add to collected victims memory
                if info['is_carrying'][0]['img_name'][8:-4] not in self._collectedVictims:
                    self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carryingTogether = True
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying']) == 0:
                self._carryingTogether = False
        # If carrying a victim together, let agent be idle (because joint actions are essentially carried out by the human)
        if self._carryingTogether == True:
            return None, {}

        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._sendMessage('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')

        # Ongoing loop untill the task is terminated, using different phases for defining the agent's behavior
        while True:
            if Phase.INTRO == self._phase:
                # Send introduction message
                self._sendMessage('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Wait untill the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._goalVic = None
                self._goalLoc = None
                self._rescue = None
                self._moving = True
                remainingZones = []
                remainingVics = []
                remaining = {}
                # Identification of the location of the drop zones
                zones = self._getDropZones(state)
                # Identification of which victims still need to be rescued and on which location they should be dropped
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                        remainingVics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remainingZones:
                    self._remainingZones = remainingZones
                    self._remaining = remaining
                # Remain idle if there are no victims left to rescue
                if not remainingZones:
                    return None, {}

                # Check which victims can be rescued next because human or agent already found them
                for vic in remainingVics:
                    # Define a previously found victim as target victim because all areas have been searched
                    if vic in self._foundVictims and vic in self._todo and len(self._searchedRooms)==0:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                        # Move to target victim
                        # ------------------
                        if "critical" in vic:
                            self._rescue = 'together'
                        if "mild" in vic:
                            if trustBeliefs.get_binary_willingness():
                                self._rescue = "together"
                            else:
                                self._rescue = "alone"
                        self._nr_skipped_action = 0
                        # ------------------
                        self._sendMessage('Moving to ' + self._foundVictimLocs[vic]['room'] + ' to pick up ' + self._goalVic +'. Please come there as well to help me carry ' + self._goalVic + ' to the drop zone.', 'RescueBot')
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # Define a previously found victim as target victim
                    if vic in self._foundVictims and vic not in self._todo:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]

                        # --------------
                        # if critical
                        if "critical" in vic:
                            # if human not willing and mildly injured, then find next victim
                            if not trustBeliefs.get_binary_willingness() and len(list(filter(lambda v: "mild" in vic, remainingVics))) > 0 and self._nr_skipped_action < self._max_allowed_skips:
                                self._nr_skipped_action += 1
                                continue
                            # if willing or only critical left, just rescue (in the hope that human will help)
                            else:
                                self._rescue = "together"
                                self._nr_skipped_action = 0
                        # when the human is weak and the victim is mildly injured
                        if 'mild' in vic and self._condition=='weak':
                            # if human is willing, rescue together
                            if trustBeliefs.get_binary_willingness():
                                self._rescue = 'together'
                            else:
                                self._rescue = "alone"
                            self._nr_skipped_action = 0
                        # ------------
                        # Rescue alone if the victim is mildly injured and the human not weak
                        if 'mild' in vic and self._condition!='weak':   # better to leave this as is, cause the human can carry multiple mildly injured victims if he decides so but should be free to do so
                            self._rescue = 'alone'
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # If there are no target victims found, visit an unsearched area to search for victims
                    if vic not in self._foundVictims or vic in self._foundVictims and vic in self._todo and len(self._searchedRooms)>0:
                        self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearchedRooms = [room['room_name'] for room in state.values()
                                   if 'class_inheritance' in room
                                   and 'Door' in room['class_inheritance']
                                   and room['room_name'] not in self._searchedRooms
                                   and room['room_name'] not in self._tosearch]
                # If all areas have been searched but the task is not finished, start searching areas again
                if self._remainingZones and len(unsearchedRooms) == 0:
                    self._tosearch = []
                    self._searchedRooms = []
                    # -------------------
                    # Instead of re-searching all areas, we should search only the rooms the human searched before
                    self._searchedRooms.extend(self._searched_rooms_by_robot)
                    # -------------------
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._sendMessage('Going to re-search all areas.', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._currentDoor == None:
                        # Find all area entrance locations
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearchedRooms, agent_location))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms, agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug (LOL)
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._currentDoor != None:
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearchedRooms, self._currentDoor))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms, self._currentDoor))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                # Switch to a different area when the human found a victim
                if self._goalVic and self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    self._doormat = state.get_room(self._foundVictimLocs[self._goalVic]['room'])[-1]['doormat']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat
                # Otherwise plan the route to the previously identified area to search
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat
                self._navigator.add_waypoints([doorLoc])
                # Follow the route to the next area to search
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                # Find the next victim to rescue if the previously identified target victim was rescued by the human
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Identify which area to move to because the human found the previously identified target victim
                if self._goalVic and self._goalVic in self._foundVictims and self._door['room_name'] != self._foundVictimLocs[self._goalVic]['room']: # TODO change trust
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Identify the next area to search if the human already searched the previously identified area
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Otherwise move to the next area to search
                else:
                    self._state_tracker.update(state)
                    # Explain why the agent is moving to the specific area, either because it containts the current target victim or because it is the closest unsearched area
                    if self._goalVic in self._foundVictims and str(self._door['room_name']) == self._foundVictimLocs[self._goalVic]['room'] and not self._remove:
                        if self._condition=='weak':
                            self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic + ' together with you.', 'RescueBot')
                        else:
                            self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic + '.', 'RescueBot')
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove :
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.', 'RescueBot')
                    self._currentDoor = self._door['location']
                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        # Remove obstacles blocking the path to the area
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4), (9, 7), (9, 19), (21, 19)]:
                                self._sendMessage('Reaching ' + str(self._door['room_name']) + ' will take a bit longer because I found stones blocking my path.', 'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    # Identify and remove obstacles if they are blocking the entrance of the area
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                # Identify which obstacle is blocking the entrance
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._sendMessage('Found rock blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + ' \n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + ' \
                                \n clock - removal time: 5 seconds \n afstand - distance between us: ' + self._distanceHuman ,'RescueBot')
                            self._waiting = True
                            self._timeStartedWaiting = datetime.now()
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        elif (self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove) or self._overrideContinue:
                            self._overrideContinue = False
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        # ------------------
                        elif self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            # if the human indicates to remove the obstacle, remove with help if trustworthy or if no other victims left
                            self._timeStartedWaiting = datetime.now()
                            if len(self._remainingZones) > 1 or self._nr_skipped_action >= self._max_allowed_skips or trustBeliefs.get_trust():
                                if not self._remove:
                                    self._answered = True
                                # Tell the human to come over and be idle untill human arrives
                                if not state[{'is_human_agent': True}]:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove rock.','RescueBot')
                                    return None, {}

                                # Tell the human to remove the obstacle when he/she arrives
                                if state[{'is_human_agent': True}]:
                                    self._sendMessage('Lets remove rock blocking ' + str(self._door['room_name']) + '!','RescueBot')
                                    return None, {}
                                self._nr_skipped_action = 0
                            else:
                                self._nr_skipped_action += 1
                                self._phase = Phase.FIND_NEXT_GOAL
                                break
                        # Remain idle untill the human communicates what to do with the identified obstacle
                        elif self._waiting:
                            if datetime.now().timestamp() - self._timeStartedWaiting.timestamp() > self._max_wait_time:
                                trustBeliefs.updateCompetence(-15/100, True)
                                self._overrideContinue = True
                                self._nr_skipped_action = 0

                            return None, {}
                        else:
                            return None, {}
                        # ------------------

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + '\n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + ' \
                                \n clock - removal time: 10 seconds','RescueBot')
                            self._waiting = True
                            self._timeStartedWaiting = datetime.now()
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        elif self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle if the human tells the agent to do so
                        elif (self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove) or self._overrideRemoveAlone:
                            self._overrideRemoveAlone = False
                            self._waiting = False
                            if not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._sendMessage('Removing tree blocking ' + str(self._door['room_name']) + '.','RescueBot')
                            if self._remove:
                                self._sendMessage('Removing tree blocking ' + str(self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remain idle untill the human communicates what to do with the identified obstacle
                        elif self._waiting:
                            if datetime.now().timestamp() - self._timeStartedWaiting.timestamp() > self._max_wait_time:
                                trustBeliefs.updateCompetence(-15/100, True)
                                self._overrideRemoveAlone = True
                            return None, {}
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + ' \n explore - areas searched: area ' + str(self._searchedRooms).replace('area','') + ' \
                                \n clock - removal time together: 3 seconds \n afstand - distance between us: ' + self._distanceHuman + '\n clock - removal time alone: 20 seconds','RescueBot')
                            self._waiting = True
                            self._timeStartedWaiting = datetime.now()
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        elif self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle alone if the human decides so
                        elif (self.received_messages_content and self.received_messages_content[-1] == 'Remove alone' and not self._remove) or self._overrideRemoveAlone:
                            self._answered = True
                            self._overrideRemoveAlone = False
                            self._waiting = False
                            self._sendMessage('Removing stones blocking ' + str(self._door['room_name']) + '.','RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remove the obstacle together if the human decides so
                        # --------------
                        elif self.received_messages_content and self.received_messages_content[-1] == 'Remove together' or self._remove:
                            # if the human indicates to remove the obstacle, remove with help if trustworthy or if no other victims left
                            self._timeStartedWaiting = datetime.now()
                            if len(self._remainingZones) > 1 or self._nr_skipped_action >= self._max_allowed_skips or trustBeliefs.get_trust():
                                if not self._remove:
                                    self._answered = True
                                # Tell the human to come over and be idle untill human arrives
                                if not state[{'is_human_agent': True}]:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove stones together.','RescueBot')
                                    return None, {}

                                # Tell the human to remove the obstacle when he/she arrives
                                if state[{'is_human_agent': True}]:
                                    self._sendMessage('Lets remove stones blocking ' + str(self._door['room_name']) + '!','RescueBot')
                                    return None, {}
                                self._nr_skipped_action = 0
                            else:
                                self._nr_skipped_action += 1
                                self._answered = True
                                self._waiting = False
                                self._sendMessage('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                                  'RescueBot')
                                self._phase = Phase.ENTER_ROOM
                                self._remove = False
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remain idle until the human communicates what to do with the identified obstacle  # TODO what if we wait too long
                        elif self._waiting:
                            if datetime.now().timestamp() - self._timeStartedWaiting.timestamp() > self._max_wait_time:
                                trustBeliefs.updateCompetence(-15/100, True)
                                self._overrideRemoveAlone = True
                            return None, {}
                        else:
                            return None, {}
                        # --------------
                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM

            if Phase.ENTER_ROOM == self._phase:
                self._answered = False
                # If the target victim is rescued by the human, identify the next victim to rescue
                if self._goalVic in self._collectedVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # If the target victim is found in a different area, start moving there # TODO, maybe adjust trust
                if self._goalVic in self._foundVictims and self._door['room_name'] != self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # If the human searched the same area, plan searching another area instead
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Otherwise, enter the area and plan to search it
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                self._agentLoc = int(self._door['room_name'].split()[-1])
                # Store the locations of all area tiles
                roomTiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._door['room_name']]
                self._roomtiles = roomTiles
                # Make the plan for searching the area
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(roomTiles))
                self._roomVics = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                # Search the area
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                trust = trustBeliefs.get_trust()
                if action != None:
                    # Identify victims present in the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            # Identify the exact location of the victim that was found by the human earlier
                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                self._recentVic = vic
                                # Add the exact victim location to the corresponding dictionary
                                self._foundVictimLocs[vic] = {'location': info['location'],'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                if vic == self._goalVic:
                                    # Communicate which victim was found
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + ' because you told me ' + vic + ' was located here.','RescueBot')
                                    # Add the area to the list with searched areas
                                    if self._door['room_name'] not in self._searchedRooms:
                                        self._searchedRooms.append(self._door['room_name'])
                                        # ----------------
                                        self._searched_rooms_by_robot.append(self._door["room_name"])
                                        # ----------------
                                    # Do not continue searching the rest of the area but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL
                            # Identify injured victim in the area
                            if 'healthy' not in vic:
                                if vic in self._foundVictims:
                                    if self._foundVictimLocs[vic]['room'] != self._door or vic in self._collectedVictims:
                                        trustBeliefs.updateWillingness(-50/100, True)
                                        self._foundVictimLocs[vic] = {'location': info['location'],'room': self._door['room_name'], 'obj_id': info['obj_id']}

                                        if vic in self._collectedVictims: # found one victim that was saved
                                            self._collectedVictims.remove(vic)
                                        
                                        trust = trustBeliefs.get_trust()

                                        if 'mild' in vic and self._answered == False and not self._waiting:
                                            if trust:
                                                self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                                    Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + '\n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + '\n \
                                                    clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distanceHuman,'RescueBot')
                                                self._waiting = True
                                                self._timeStartedWaiting = datetime.now()
                                            else:
                                                self._overrideRescueAlone = True
                                                self._waiting = True

                                        if 'critical' in vic and self._answered == False and not self._waiting:
                                            self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                                Important features to consider are: \n explore - areas searched: area ' + str(self._searchedRooms).replace('area','') + ' \n safe - victims rescued: ' + str(self._collectedVictims) + '\n \
                                                afstand - distance between us: ' + self._distanceHuman,'RescueBot')
                                            self._timeStartedWaiting = datetime.now()
                                            self._waiting = True

                                        
                                if vic not in self._foundVictims:
                                    self._recentVic = vic
                                    # Add the victim and the location to the corresponding dictionary
                                    self._foundVictims.append(vic)
                                    self._foundVictimLocs[vic] = {'location': info['location'],'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                    # Communicate which victim the agent found and ask the human whether to rescue the victim now or at a later stage
                                    

                                    if vic in self._collectedVictims: # found one victim that was saved
                                        self._collectedVictims.remove(vic)
                                        trustBeliefs.updateWillingness(-50/100, True) 


                                    if 'mild' in vic and self._answered == False and not self._waiting:
                                        # print(trust)
                                        if trust:
                                            self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + '\n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + '\n \
                                                clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distanceHuman,'RescueBot')
                                            self._waiting = True
                                            self._timeStartedWaiting = datetime.now()
                                        else:
                                            self._overrideRescueAlone = True
                                            self._waiting = True

                                    if 'critical' in vic and self._answered == False and not self._waiting:
                                        self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                            Important features to consider are: \n explore - areas searched: area ' + str(self._searchedRooms).replace('area','') + ' \n safe - victims rescued: ' + str(self._collectedVictims) + '\n \
                                            afstand - distance between us: ' + self._distanceHuman,'RescueBot')
                                        self._timeStartedWaiting = datetime.now()
                                        self._waiting = True

                    # Execute move actions to explore the area
                    return action, {}
                

                #TODO FOLLOW_ROOM_SEARCH_PATH: take into account trust when interpreting the message from human
                #TODO FOLLOW_ROOM_SEARCH_PATH: don't wait too long until making a decision (at some point do what you think it should be done)
                # Communicate that the agent did not find the target victim in the area while the human previously communicated the victim was located here
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room'] == self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic + '.','RescueBot')
                    # Remove the victim location from memory
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searchedRooms:
                    self._searchedRooms.append(self._door['room_name'])
                    # ----------------
                    self._searched_rooms_by_robot.append(self._door["room_name"])
                    # ----------------


                # Make a plan to rescue a found critically injured victim if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue' and 'critical' in self._recentVic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    self._timeStartedWaiting = datetime.now()
                    # Tell the human to come over and help carry the critically injured victim
                    if not state[{'is_human_agent': True}]:
                        self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together.', 'RescueBot')
                    # Tell the human to carry the critically injured victim together
                    if state[{'is_human_agent': True}]:
                        self._sendMessage('Lets carry ' + str(self._recentVic) + ' together! Please wait until I moved on top of ' + str(self._recentVic) + '.', 'RescueBot')
                    self._goalVic = self._recentVic
                    self._recentVic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM

                # Make a plan to rescue the mildly injured victim alone if the human decides so, and communicate this to the human
                if (self.received_messages_content and self.received_messages_content[-1] == 'Rescue alone' or self._overrideRescueAlone) and 'mild' in self._recentVic:
                    self._sendMessage('Picking up ' + self._recentVic + ' in ' + self._door['room_name'] + '.','RescueBot')
                    self._rescue = 'alone'
                    self._overrideRescueAlone = False
                    self._answered = True
                    self._waiting = False
                    # ------------ Some new bug fixes from original creator
                    self._goalVic = self._recentVic
                    self._goalLoc = self._remaining[self._goalVic]
                    self._recentVic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                
                # Make a plan to rescue a found mildly injured victim together if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue together' and 'mild' in self._recentVic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    self._timeStartedWaiting = datetime.now()
                    # Tell the human to come over and help carry the mildly injured victim
                    if not state[{'is_human_agent': True}]:
                        self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together.', 'RescueBot')
                    # Tell the human to carry the mildly injured victim together
                    if state[{'is_human_agent': True}]:
                        self._sendMessage('Lets carry ' + str(self._recentVic) + ' together! Please wait until I moved on top of ' + str(self._recentVic) + '.', 'RescueBot')
                    self._goalVic = self._recentVic
                    self._recentVic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM

                    # -------------
                # Continue searching other areas if the human decides so
                if (self.received_messages_content and self.received_messages_content[-1] == 'Continue') or self._overrideContinue:
                    self._answered = True
                    self._overrideContinue = False
                    self._waiting = False
                    self._todo.append(self._recentVic)
                    self._recentVic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Remain idle untill the human communicates to the agent what to do with the found victim
                if self.received_messages_content and self._waiting and self.received_messages_content[-1] != 'Rescue' and self.received_messages_content[-1] != 'Continue':
                    if datetime.now().timestamp() - self._timeStartedWaiting.timestamp() >= self._max_wait_time:
                        trustBeliefs.updateCompetence(-15/100, True)
                        if 'mild' in self._recentVic:
                            self._overrideRescueAlone = True
                        elif 'critical' in self._recentVic:
                            self._overrideContinue = True
                    return None, {}
                # Find the next area to search when the agent is not waiting for an answer from the human or occupied with rescuing a victim
                if not self._waiting and not self._rescue:
                    self._recentVic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}




            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                # Plan the path to a found victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                # Follow the path to the found victim
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # Start searching for other victims if the human already rescued the target victim
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._phase = Phase.FIND_NEXT_GOAL
                # Otherwise, move towards the location of the found victim
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                # Store all area tiles in a list
                roomTiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._foundVictimLocs[self._goalVic]['room']]
                self._roomtiles = roomTiles
                objects = []
                # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                for info in state.values():
                    # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                        'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles and self._rescue=='together' or \
                        self._goalVic in self._foundVictims and self._goalVic in self._todo and len(self._searchedRooms)==0 and 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                        self._goalVic in self._foundVictims and self._goalVic in self._todo and len(self._searchedRooms)==0 and 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        # Remain idle when the human has not arrived at the location
                        if not self._humanName in info['name']:
                            if datetime.now().timestamp() - self._timeStartedWaiting.timestamp() >= self._max_wait_time:
                                trustBeliefs.updateCompetence(-15/100, True)
                                self._waiting = False
                                self._phase = Phase.FIND_NEXT_GOAL
                                return None, {}
                            self._waiting = True
                            self._moving = False
                            return None, {}
                # Add the victim to the list of rescued victims when it has been picked up

                if len(objects) == 0 and 'critical' in self._goalVic or len(objects) == 0 and 'mild' in self._goalVic and self._rescue=='together':
                    self._waiting = False
                    if self._goalVic not in self._collectedVictims:
                        self._collectedVictims.append(self._goalVic)
                    self._carryingTogether = True
                    # Determine the next victim to rescue or search
                    self._phase = Phase.FIND_NEXT_GOAL
                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone
                if 'mild' in self._goalVic and self._rescue=='alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goalVic not in self._collectedVictims:
                        self._collectedVictims.append(self._goalVic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._foundVictimLocs[self._goalVic]['obj_id'], 'human_name':self._humanName}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goalLoc])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                # Communicate that the agent is transporting a mildly injured victim alone to the drop zone
                if 'mild' in self._goalVic and self._rescue=='alone':
                    self._sendMessage('Transporting ' + self._goalVic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                # Follow the path to the drop zone
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # Drop the victim at the drop zone
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # TODO add a phase to check what other victims are dropped and update the dropped list
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._goalVic and self._rescue=='alone':
                    self._sendMessage('Delivered ' + self._goalVic + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._currentDoor = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__, {'human_name': self._humanName}

    def _getDropZones(self, state):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones


    def _processMessages(self, state, teamMembers, condition, trustBeliefs: TrustBelief):
        '''
        process incoming messages received from the team members
        '''

        # process incoming messages received from the team members
        
        receivedMessages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        # Check the content of the received messages
        for mssgs in receivedMessages.values():     # TODO remove messages afterwards. Needed?
            for msg in mssgs:
                # -----------------
                # If a received message involves team members searching areas, add these areas to the memory of areas that have been explored, if the human is trustworthy (both competent and willing)
                if msg.startswith("Search:") and trustBeliefs.get_trust():
                    area = 'area ' + msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                    else: # if room was marked twice as searched
                        trustBeliefs.updateWillingness(-20/100)

                # TODO is human competent?
                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:") and trustBeliefs.get_binary_willingness():
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    # Add the victim and its location to memory
                    if foundVic not in self._foundVictims:
                        self._foundVictims.append(foundVic)
                        self._foundVictimLocs[foundVic] = {'room': loc}
                    elif foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                        self._foundVictimLocs[foundVic] = {'room': loc}
                        trustBeliefs.updateWillingness(-20/100) # if the first time the human gave a wrong direction
                    # elif foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] == loc:
                        

                    # Decide to help the human carry a found victim when the human's condition is 'weak'
                    if condition=='weak':
                        self._rescue = 'together'
                    # Add the found victim to the to do list when the human's condition is not 'weak'
                    if 'mild' in foundVic and condition!='weak':
                        self._todo.append(foundVic)
                # If a received message involves team members rescuing victims, add these victims and their locations to memory, if the human is trustworthy (both competent and willing)
                if msg.startswith('Collect:') and trustBeliefs.get_trust():
                    # Identify which victim and area it concerns
                    trustBeliefs.updateCompetence(25/100) # update competence because the human was supposedly able to carry a victim
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    # Add the victim and location to the memory of found victims
                    if collectVic not in self._foundVictims:
                        self._foundVictims.append(collectVic)
                        self._foundVictimLocs[collectVic] = {'room': loc}
                    if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                        self._foundVictimLocs[collectVic] = {'room': loc}
                    # Add the victim to the memory of rescued victims when the human's condition is not weak
                    if condition!='weak' and collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                    # Decide to help the human carry the victim together when the human's condition is weak
                    if condition=='weak':
                        self._rescue = 'together'
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over, if the human is trustworthy (both competent and willing)
                if msg.startswith('Remove:') and trustBeliefs.get_trust():
                    # Come over immediately when the agent is not carrying a victim
                    if not self._carrying:
                        # Identify at which location the human needs help
                        area = 'area ' + msg.split()[-1]
                        self._door = state.get_room_doors(area)[0]
                        self._doormat = state.get_room(area)[-1]['doormat']
                        if area in self._searchedRooms: # TODO update trust (competency) as they lied before
                            self._searchedRooms.remove(area)
                        # Clear received messages (bug fix)
                        self.received_messages = []
                        self.received_messages_content = []
                        self._moving = True
                        self._remove = True
                        if self._waiting and self._recentVic:
                            self._todo.append(self._recentVic)
                        self._waiting = False
                        # Let the human know that the agent is coming over to help
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.','RescueBot')
                        # Plan the path to the relevant area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Come over to help after dropping a victim that is currently being carried by the agent
                    # -----------------
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._sendMessage('Will come to ' + area + ' after dropping ' + self._goalVic + '.','RescueBot')
            # Store the current location of the human in memory
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']:
                self._humanLoc = int(mssgs[-1].split()[-1])

            trustBeliefs.flushUpdates()


    def _trustBelief(self, members, trustBeliefs: TrustBelief, folder, receivedMessages):
        '''
        TO DO ( Vlad ):
            -> [BUG] some removal have "Remove togheter", some do not, handle this
        OBSERVATIONS ( Vlad ):
            ->Trust is much harder to gain than to lose as mistakes cost more time than 
                we benefit if we collaborate ( one mistakes still weight more than one right action ) 
            ->Willingness can be wierd because there might be motive behind it, but not enough time/methods to communicate it
        '''

       
        # Update the trust value based on for example the received messages
                                                            # TO DO: useless for loop
        for message in self._sendMessages:

            # reduce 20% of the competence score for misscommunication of victims
            if 'because I searched the whole area without finding' in message:                  # subtract the same competence for critical and normal?
                trustBeliefs.updateCompetence(-20/100)
                self._sendMessages.remove(message)   
            
            # add 15% of the competence score for helpfull communication of victims
            if 'because you told me' in message and 'was located here' in message:
                trustBeliefs.updateWillingness(15/100)
                self._sendMessages.remove(message)   

            if 'blocking area' in message and 'Please decide whether to' in message:
                self.waitingForDecisionResponse = True
                self.decisionDistance = message.split('distance between us:')[1]  
                self._sendMessages.remove(message)      

            # The human has asked help to remove something but the human could also have done itself,
            # however the human has not lied about the obstacle
            if 'because you asked me to' in message:
                # trustBeliefs[self._humanName]['competence']+= np.clip(trustBeliefs[self._humanName]['competence']/100 * 15, 0, 1)
                # trustBeliefs[self._humanName]['willingness']-= np.clip(trustBeliefs[self._humanName]['willingness']/100 * 10, 0, 1)
                trustBeliefs.updateCompetence(15/100)
                trustBeliefs.updateWillingness(-10/100)
                self._sendMessages.remove(message)


            # The human has lied about something since all the rooms have been
            # searched and some victims are still not found
            if 're-search' in message:
                # trustBeliefs[self._humanName]['competence'] -= np.clip(trustBeliefs[self._humanName]['competence']/100 * 25, 0, 1)
                trustBeliefs.updateCompetence(-25/100)
                self._sendMessages.remove(message)

            # The human has asked help to remove something it could not do itself
            if 'Lets remove' in message:
                # trustBeliefs[self._humanName]['competence']+=np.clip(trustBeliefs[self._humanName]['competence']/100 * 15, 0, 1)
                # trustBeliefs[self._humanName]['willingness']+=np.clip(trustBeliefs[self._humanName]['willingness']/100 * 15, 0, 1)
                trustBeliefs.updateCompetence(15/100)
                trustBeliefs.updateWillingness(15/100)
                self._sendMessages.remove(message)


        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # Update the trust value based on for example the received messages
        for message in receivedMessages:
            # The human does not want the robot to rescue a victim
            if 'Continue' in message:
                if self.waitingForDecisionResponse:                        # We can collaborate but you refused, reduce willingness
                    trustChangeValue = 30/100 if "close" in self.decisionDistance else 20/100 
                    trustBeliefs.updateWillingness(-trustChangeValue)
                    self.decisionDistance = None
                    self.waitingForDecisionResponse = False
                else:
                    trustBeliefs.updateWillingness(-5/100)

            if 'Remove alone' in message:
                if self.waitingForDecisionResponse:                        # We can collaborate but you decided you do not want to help
                    trustChangeValue = 20/100 if "close" in self.decisionDistance else 10/100 
                    trustBeliefs.updateWillingness(-trustChangeValue)
                    self.decisionDistance = None
                    self.waitingForDecisionResponse = False

                receivedMessages.remove(message)

            # The human is willing to help the robot
            if 'Remove together' in message:
                # trustBeliefs[self._humanName]['willingness'] += np.clip(trustBeliefs[self._humanName]['willingness']/100 * 15, 0 , 1)
                if self.waitingForDecisionResponse:                         # We can collaborate and you want to collaborate
                    trustChangeValue = 15/100 if "close" in self.decisionDistance else 20/100 
                    trustBeliefs.updateWillingness(trustChangeValue)
                    self.decisionDistance = None
                    self.waitingForDecisionResponse = False
                else:
                    trustBeliefs.updateWillingness(15/100)

            # The human is willing to help the robot rescue a mildly injured victim
            if 'Rescue together' in message and self._goalVic and 'mild' in self._goalVic:
                # trustBeliefs[self._humanName]['willingness'] += np.clip(trustBeliefs[self._humanName]['willingness']/100 * 15, 0 , 1)
                trustBeliefs.updateWillingness(15/100)

            # The human is willing to help the robot rescue a critically injured victim
            if 'Rescue' in message and self._goalVic and 'critical' in self._goalVic:
                # trustBeliefs[self._humanName]['willingness'] += np.clip(trustBeliefs[self._humanName]['willingness']/100 * 15, 0 , 1)
                trustBeliefs.updateWillingness(15/100)

            if 'Remove' in message:                                                 # Only good option if we cannot work togheter
                #receivedMessages.remove(message)
                self.waitingForDecisionResponse = False

            # Increase agent trust in a team member that rescued a victim
            #if 'Collect' in message:
            #    trustBeliefs[self._humanName]['competence']+=0.10
                # Restrict the competence belief to a range of -1 to 1
            #    trustBeliefs[self._humanName]['competence'] = np.clip(trustBeliefs[self._humanName]['competence'], -1, 1)

        trustBeliefs.flushUpdates()
        # Save current trust belief values so we can later use and retrieve them to add to a csv file with all the logged trust belief values
        # with open(folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
        #     csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(['name','competence','willingness'])
        #     csv_writer.writerow([self._humanName,trustBeliefs[self._humanName]['competence'],trustBeliefs[self._humanName]['willingness']])

        return trustBeliefs

    def _sendMessage(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        '''
        calculate which area is closest to the agent's location
        '''
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        '''
        efficiently transverse areas instead of moving over every single area tile
        '''
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs
