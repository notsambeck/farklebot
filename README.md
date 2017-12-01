# farklebot
(_n. a bot that learns to play the dice game Farkle_)

### About Farkle:
[Farkle](https://en.m.wikipedia.org/wiki/Farkle) is a dice game that has existed for a long time, although you can also pay money to buy it in a cool box that makes the name sound ironic.

### About farklebot
This repository implements an algorithm that learns to play the game.

It learns based on playing many times while adjusting strategy. While it would be possible and more computationally efficient to 'solve' the game based on conditional probabilities for each roll, I found it more interesting to have the system learn independently.

At present the choice of dice is hard-coded; the learning applies to deciding when to continue or stop a turn (points are scored only when you stop before a non-scoring roll). The system outputs its decision boundary when it has converged to an approximate solution.

Building an algorithm (e.g. a neural net) that selects dice to keep (after each roll, before the above decision to continue or stop) would be a somewhat interesting, but that might be more complex than this project really deserves. It would also allow for optimization of that process, which would probably improve the mean score.

### Conclusions:
1. Gradients in this kind of a non-continous space (score for a game) are not always well behaved, even for very large sample sizes.
2. Accurately evaluating different strategies takes an extremely large number of samples; sub-optimal strategies will frequently pay off.
3. Corroloary: In practice, a reasonanble guess at a good strategy may be just as good as the optimal strategy.
