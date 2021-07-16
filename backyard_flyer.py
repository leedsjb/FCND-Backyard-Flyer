import argparse
import time
from enum import Enum
import traceback

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID

class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5

class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = [0.0, 0.0, 0.0]
        self.next_wpt = 0
        self.all_waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.debug = True

        # initial state
        self.flight_state = States.MANUAL

        # TODO: Register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        """
        Triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data

        :return: void
        """

        # determine if takeoff is complete, if so begin flightplan
        if self.flight_state == States.TAKEOFF:

            # coordinate conversion (because + down is used)
            altitude = -1.0 * self.local_position[2]

            # check if altitude is within 95% of target
            if altitude > (0.95 * self.target_position[2]):
                self.waypoint_transition()

        # determine if current waypoint has been reached
        # if so, update the waypoint to the next waypoint in the flightplan
        
        FLT_PLAN_WPTS = len(self.all_waypoints)

        if(self.flight_state == States.WAYPOINT and self.next_wpt < FLT_PLAN_WPTS and self.is_wpt_reached()):
            self.waypoint_transition()
         
        elif(self.flight_state == States.WAYPOINT and self.next_wpt == FLT_PLAN_WPTS and self.is_wpt_reached()):
            self.landing_transition()
        
    def velocity_callback(self):
        """
        Triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data

        :return: void
        """
        drone_landed = abs(self.local_position[2]) < 0.1

        if self.flight_state == States.LANDING and drone_landed: # ensure drone has landed
                self.disarming_transition()

    def state_callback(self):
        """
        Triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data

        :return: void
        """

        if not self.in_mission:
            return
        elif self.flight_state == States.MANUAL:
            self.arming_transition()
        elif self.flight_state == States.ARMING:
            self.takeoff_transition()
        elif self.flight_state == States.DISARMING:
            self.manual_transition()
   

    def calculate_box(self):
        """Calculates the waypoints for a box shaped flightplan
        
        :return: numpy array. an array of lists, each list is a waypoint
        """

        # calculate a right-hand turn path in the shape of a box
        
        BOX_LEG_LENGTH = 10.0 # 10.0 meters per leg (side) of the box
        CRUISING_ALT = 3.0 # the altitude for each waypoint

        start_N_pos = self.local_position[0]
        start_E_pos = self.local_position[1]
        current_N_pos = self.local_position[0]
        current_E_pos = self.local_position[1]

        # create local array of objects to hold 4 waypoints
        waypoints = np.empty(4, dtype=object)
        
        # move north BOX_LEG_LENGTH
        current_N_pos += BOX_LEG_LENGTH
        waypoints[0] = [current_N_pos, current_E_pos, CRUISING_ALT]

        # move east BOX_LEG_LENGTH
        current_E_pos += BOX_LEG_LENGTH
        waypoints[1] = [current_N_pos, current_E_pos, CRUISING_ALT]

        # move south BOX_LEG_LENGTH
        current_N_pos -= BOX_LEG_LENGTH
        waypoints[2] = [current_N_pos, current_E_pos, CRUISING_ALT]

        # return to start position
        waypoints[3] = [start_N_pos, start_E_pos, CRUISING_ALT]
        
        if self.debug:
            print("the flightplan is: ")
            for wp in waypoints:
                print(wp)

        return waypoints

    def arming_transition(self):
        """Arms the drone and sets the home location to the current position

        Steps:
        1. Take control of the drone
        2. Pass an arming command
        3. Set the home location to current position
        4. Transition to the ARMING state

        :return: void
        """

        if self.debug:
            print("arming transition")

        self.take_control()
        self.arm()

        homeN = self.global_position[0]
        homeE = self.global_position[1]
        homeD = self.global_position[2]

        if self.debug:
            print("setting home position: ")
            print("[N: {0}, E: {1}, D: {2}]".format(homeN, homeE, homeD))

        self.set_home_as_current_position()

        # arming transition complete, set state to ARMING
        # to trigger takeoff
        self.flight_state = States.ARMING

    def takeoff_transition(self):
        """Commands the drone to takeoff to a fixed target altitude
        of 3.0m
        
        Steps:
        1. Set target_position altitude to 3.0m
        2. Command a takeoff to 3.0m
        3. Transition to the TAKEOFF state

        :return: void
        """

        if self.debug:
            print("takeoff transition")
            print("the local position is: N: {0}, E: {1}, D: {2}".format(self.local_position[0], self.local_position[1], self.local_position[2]))

        # calculate box waypoints and set drone's self.all_waypoints
        self.all_waypoints = self.calculate_box()

        TARGET_ALTITUDE = 3.0
        self.target_position = [self.local_position[0],
                                self.local_position[1],
                                TARGET_ALTITUDE]
        
        if self.debug:
            print("the target position is: {0}".format(self.target_position))
        
        self.takeoff(TARGET_ALTITUDE)

        # takeoff in progress, set state to TAKEOFF to begin 
        # flight via waypoints in flightplan
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        """Commands drone to fly to the next waypoint in the flightplan
        Invariant: Must only be called after takeoff is complete

        1. Command the next waypoint position
        2. Transition to WAYPOINT state

        :return: void
        """
        
        # set waypoint
        if self.debug:
            print("waypoint transition")
            print("the next waypoint is wpt #: {0}".format(self.next_wpt))
        
        self.target_position = self.all_waypoints[self.next_wpt]

        if self.debug:
            print("next waypoint set:")
            print("the target position is: {0}".format(self.target_position))
        
        # command waypoint
        self.cmd_position(self.target_position[0],
                          self.target_position[1],
                          self.target_position[2],
                          0)
        
        # increment waypoint to next waypoint in flight plan
        # for next time method is called
        self.next_wpt = self.next_wpt + 1

        # if first waypoint in a flight ensure flight_state is updated
        if(self.flight_state == States.TAKEOFF):
            self.flight_state = States.WAYPOINT
       
  
    def is_wpt_reached(self):
        """
        Determines if the current local_position
        is equal to (within a reasonable tolerance) 
        the target_position. Returns boolean true if so

        :return: boolean, a boolean that's true if waypoint has been reached
        """

        # drone must land < 1m from desired position
        POS_TOL = 0.1 # the maximum error of the position
        
        # calculate position deltas for NED coor frame
        deltaN = abs(self.local_position[0] - self.target_position[0])
        deltaE = abs(self.local_position[1] - self.target_position[1])
        
        # multiply altitude by -1.0 because NED coor frame is positive down
        deltaD = abs(-1.0 * self.local_position[2] - self.target_position[2])
        
        if self.debug:
            print("nDelta: " + str(deltaN) +
                  ", eDelta: " + str(deltaE) +
                  ", dDelta: " + str(deltaD))

        # wait to transition states until desired position is reached
        if (deltaN < POS_TOL and deltaE < POS_TOL and deltaD < POS_TOL):
            return True
        else:
            return False

    def landing_transition(self):
        """Lands the drone and updates self.flight_state
        
        1. Command the drone to land
        2. Transition to the LANDING state

        :return: void
        """

        if self.debug:
            print("landing transition")

        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        """Disarms the drone and updates self.flight_state
        
        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """

        if self.debug:
            print("disarm transition")
        self.disarm()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        """Returns drone to manual control and updates self.flight_state
        
        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """

        if self.debug:
            print("manual transition")

        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        """Opens a log file and starts connection to drone
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()
