# farklebot
(a bot that learns to play the dice game Farkle)

### About Farkle:
Farkle[https://en.m.wikipedia.org/wiki/Farkle] is a dice game that has existed for a long time.

### About farklebot
This repository implements an algorithm that learns to play the game.

It learns based on playing many times while adjusting strategy. While it would be possible and more computationally efficient to simply solve the game based on conditional probabilities for each roll, I found it more interesting to have the system learn indepnently.

At present the choice of dice is hard-coded; the learning applies to deciding when to continue or stop a turn (points are scored only when you stop before a non-scoring roll). The system outputs its decision boundary when it has converved.

Building an algorithm (neural net?) that selects dice to keep (after rolling for that turn) would be a somewhat interesting, but that might be more complex than this kind of silly project really deserves. It would also allow for optimization of that process (dependent on context).

### Conclusions:
1. Gradients in this kind of a non-continous space (score for a game) are not always well behaved.
2. Evaluating the value of different strategies takes an extremely large number of samples; sub-optimal strategies will frequently pay off, just not as much as the optimal strategy.
