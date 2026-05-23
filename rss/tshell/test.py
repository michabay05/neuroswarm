from swarmsim.world.RectangularWorld import RectangularWorld, RectangularWorldConfig
from swarmsim.world.spawners.AgentSpawner import PointAgentSpawner
from swarmsim.agent.MazeAgent import MazeAgent, MazeAgentConfig
from swarmsim.world.simulate import main as sim
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.BinaryController import BinaryController
from swarmsim.agent.control.HumanController import HumanController
from swarmsim.agent.control.StaticController import StaticController

V_MAX, W_MAX = 0.27, 0.6
CENTER_POS = (5, 5)

# # world
# world_config = RectangularWorldConfig(size=(10, 10), time_step=1 / 40)
# world = RectangularWorld(world_config)
#
# # control = HumanController(speed_range=(-V_MAX, V_MAX), turn_range=(-W_MAX, W_MAX))
# control = StaticController(output=(0.0, 0.0))
# mario = MazeAgent(
#     MazeAgentConfig(position=CENTER_POS, agent_radius=0.1, controller=control), world)
# world.population.append(mario)
#
# # spawned binary controller agent
# controller = BinaryController(a=(V_MAX, -W_MAX), b=(V_MAX, W_MAX))
# turtle_model = MazeAgent(
#     MazeAgentConfig(position=(5, 5), agent_radius=0.1, controller=controller), world)
# sensor = BinaryFOVSensor(turtle_model, theta=0.45, distance=1, bias=0)
# turtle_model.sensors.append(sensor)
#
# # spawner
# spawner = PointAgentSpawner(
#     world, n=6, facing="away", avoid_overlap=True, agent=turtle_model, mode="oneshot")
# world.spawners.append(spawner)

rwc = RectangularWorldConfig.from_yaml("world.yaml")
world = RectangularWorld(rwc)

sim(world, start_paused=True)
