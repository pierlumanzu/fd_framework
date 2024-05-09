from problems.instances.jos_1 import JOS_1
from problems.instances.mmr_5 import MMR_5


PROBLEMS = {'JOS': [JOS_1], 'MMR': [MMR_5]}

dimensions = [
    2, 3, 4, 5, 6, 8, 10, 12, 15, 17,
    20, 25, 30, 35, 40, 45, 50, 100, 200
]

PROBLEM_DIMENSIONS = {'JOS': dimensions, 'MMR': dimensions}
