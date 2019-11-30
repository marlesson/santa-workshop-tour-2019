
import random
import numpy
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

N_DAYS        = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
FAMINY_SIZE   = 5000
DAYS          = list(range(N_DAYS,0,-1))

#creator = None
def set_creator(cr):
    global creator
    creator = cr

set_creator(creator)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def cost_function(prediction, family_size_ls, choice_dict, choice_dict_num, penalties_dict):
    penalty = 0
    #print(prediction, days, family_size_ls, choice_dict_num, choice_dict)

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in DAYS}
    
    # Looping over each family; d is the day, n is size of that family, 
    # and choice is their top choices
    for n, c, c_dict, choice in zip(family_size_ls, prediction, list(choice_dict.values()), choice_dict_num):
        
        d = int(c)
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d not in choice:
            penalty += penalties_dict[n][-1]
        else:
            penalty += penalties_dict[n][choice[d]]

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    k = 0
    for v in daily_occupancy.values():
        if (v > MAX_OCCUPANCY):
            k = k + (v - MAX_OCCUPANCY)
        if (v < MIN_OCCUPANCY):
            k = k + (MIN_OCCUPANCY - v)
    #    if k > 0:
    #        penalty += 100000000 
    penalty += 100000*k

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[DAYS[0]]-125.0) / 400.0 * daily_occupancy[DAYS[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[DAYS[0]]
    for day in DAYS[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return (penalty, )

def main(args):
  print(args)
  print("")

  #load dataset
  data       = pd.read_csv('data/family_data.csv', index_col='family_id')
  submission = pd.read_csv('data/sample_submission.csv', index_col='family_id')

  print(data.head())
  print(data.shape)

  # Load util variables
  family_size_dict  = data[['n_people']].to_dict()['n_people']
  cols              = [f'choice_{i}' for i in range(10)]
  choice_dict       = data[cols].T.to_dict()

  # from 100 to 1
  family_size_ls  = list(family_size_dict.values())
  choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]

  # Computer penalities in a list
  penalties_dict = {
      n: [
          0,
          50,
          50 + 9 * n,
          100 + 9 * n,
          200 + 9 * n,
          200 + 18 * n,
          300 + 18 * n,
          300 + 36 * n,
          400 + 36 * n,
          500 + 36 * n + 199 * n,
          500 + 36 * n + 398 * n
      ]
      for n in range(max(family_size_dict.values())+1)
  } 

  # AG
  # --------------------------------------------------
  toolbox = base.Toolbox()
  pool    = multiprocessing.Pool()
  # Geral
  toolbox.register("map", pool.map)

  # Attribute generator
  toolbox.register("attr_int",   random.randint, 1, 100)

  # Structure initializers
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, FAMINY_SIZE)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  toolbox.register("evaluate",   cost_function, family_size_ls=family_size_ls, choice_dict=choice_dict, 
                                                  choice_dict_num=choice_dict_num, penalties_dict=penalties_dict)
  toolbox.register("mate",       tools.cxUniform, indpb=0.1)
  toolbox.register("mutate",     tools.mutUniformInt, low=1, up=100, indpb=args.mutpb2)
  toolbox.register("select",     tools.selNSGA2)

  # Params 
  MU, LAMBDA = args.npop*3, args.npop

  # Population
  pop   = toolbox.population(n=MU)
  hof   = tools.ParetoFront()
  stats = tools.Statistics(lambda ind: ind.fitness.values)

  # Statistics
  stats.register("avg", numpy.mean, axis=0)
  stats.register("std", numpy.std, axis=0)
  stats.register("min", numpy.min, axis=0)
  stats.register("max", numpy.max, axis=0)

  pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                          cxpb=args.cxpb,   mutpb=args.mutpb, ngen=args.ngen, 
                                          stats=stats, halloffame=hof)


  # Best Solution
  best_solution = tools.selBest(pop, 1)[0]
  print("")
  print("[{}] best_score: {}".format(logbook[-1]['gen'], logbook[-1]['min'][0]))

  with open('results/bestscore.txt', 'w') as f:
    f.write(str(logbook[-1]['min'][0]))

  with open('results/logbook.txt', 'w') as f:
    f.write(str(logbook))

  daily_occupancy = get_daily_occupancy(best_solution, family_size_ls, choice_dict_num) 

  save_submission(submission, best_solution)
  save_viz(logbook, daily_occupancy)

def save_viz(logbook, daily_occupancy):
  # PLot curve
  plt.figure(figsize=(10,8))
  front = np.array([(c['gen'], c['avg'][0]) for c in logbook])
  plt.plot(front[:,0][1:-1], np.log(front[:,1][1:-1]), "-bo", c="b")
  plt.axis("tight")
  plt.savefig('results/history.png')  

  # PLot Ocuppancy
  plt.figure(figsize=(10,8))
  plt.hlines(MAX_OCCUPANCY, 0, 100, color='r')
  plt.plot(list(daily_occupancy.values()), 'bo')
  plt.plot(list(daily_occupancy.values()))
  plt.hlines(MIN_OCCUPANCY, 0, 100, color='r')
  plt.savefig('results/daily_occupancy.png')

def get_daily_occupancy(solution, family_size_ls, choice_dict_num):
  # We'll use this to count the number of people scheduled each day
  daily_occupancy = {k:0 for k in DAYS}

  # Looping over each family; d is the day, n is size of that family, 
  # and choice is their top choices
  for n, d, choice in zip(family_size_ls, solution, choice_dict_num):
      # add the family member count to the daily occupancy
      daily_occupancy[d] += n
  
  return daily_occupancy

def save_submission(submission, best_solution):
  print("save_submission")
  days = []
  for c in best_solution:
    days.append(c)

  submission['assigned_day']=days
  print(submission.head())
  submission.to_csv('submission.csv')  

# Params
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Params
    parser.add_argument('--ngen', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')

    parser.add_argument('--npop', type=int, default=100, metavar='N',
                        help='')

    parser.add_argument('--mutpb2', type=float, default=0.05, metavar='N',
                        help='Mutation per crommoso')

    parser.add_argument('--mutpb', type=float, default=0.3, metavar='N',
                        help='Mutation per individuo')

    parser.add_argument('--cxpb', type=float, default=0.7, metavar='N',
                        help='')                        

    main(parser.parse_args())    