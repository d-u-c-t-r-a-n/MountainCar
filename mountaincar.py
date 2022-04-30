import time
import gym
import numpy as np
import time


def initQTable(degOfEst, numVar, numAction):
    # [degOfEst, degOfEst, degOfEst, degOfEst, numAction]
    dimension = [degOfEst]*numVar + [numAction]
    qTable = np.zeros(dimension)
    return qTable


# listOfCriteria[0]   = degOfEst
# listOfCriteria[1]   = (lowerRange1, upperRange1)
# listOfCriteria[2]   = (lowerRange2, upperRange2)
# listOfCriteria[...] = so on
def getApproxValues(listOfCriteria):
    approxValues = list()
    degOfEst = listOfCriteria[0]
    for i in range(1, len(listOfCriteria)):
        approxValues.append(np.linspace(
            listOfCriteria[i][0], listOfCriteria[i][1], degOfEst))
    return approxValues


def getApproximation(observation, approxValues):
    qTableIndex = list()
    for i in range(len(observation)):
        qTableIndex.append(np.digitize(observation[i], approxValues[i])-1)
    return tuple(qTableIndex)


def main():
    start = time.time()
    env = gym.make('MountainCar-v0')

    degOfEst = 35
    numVar = 2
    numAction = 3
    episode = 10000
    qTable = initQTable(degOfEst, numVar, numAction)

    alpha = 0.1
    gamma = 0.98
    epsilon = 0.999
    epsilonDecay = 0.0001

    criteria = [degOfEst, (-2, 1), (-0.1, 0.1)]
    approxValues = getApproxValues(criteria)

    # Learning starts
    for _ in range(episode):
        observation = env.reset()
        qTableIndex = getApproximation(observation, approxValues)

        done = False
        action = float("inf")

        while not done:
            if np.random.random() > epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qTable[qTableIndex])

            newObservation, reward, done, _ = env.step(action)

            if newObservation[0] >= 0.5:
                reward += 1000

            newQTableIndex = getApproximation(newObservation, approxValues)

            qTable[qTableIndex][action] = alpha * (reward + gamma *
                                                   np.max(qTable[newQTableIndex])) + (1-alpha) * qTable[qTableIndex][action]

            qTableIndex = newQTableIndex
        epsilon -= epsilonDecay

    end = time.time()
    print("### Train time = " + str(end-start))

    # Test run starts
    finalAvg = list()
    for _ in range(10):
        start = time.time()
        done = False
        observation = env.reset()
        qTableIndex = getApproximation(observation, approxValues)
        while not done:
            env.render()
            action = np.argmax(qTable[qTableIndex])
            newObservation, reward, done, _ = env.step(action)
            qTableIndex = getApproximation(newObservation, approxValues)
        end = time.time()
        finalAvg.append(end-start)
    print("### Avg time = "+(str(sum(finalAvg)/len(finalAvg)))+" ###")


if __name__ == "__main__":
    main()
